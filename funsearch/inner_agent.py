"""ReAct inner-loop agent that proposes lens model parameters.

The agent receives an observed image and a set of reference proposals from
the database, then iteratively evaluates and refines its own proposals
using the ``evaluate`` tool.  It communicates via ``<action>`` blocks;
all freeform text is kept in context.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm_client import (OpenRouterClient, ContextLengthExceeded,
                         BudgetExhausted, OpenRouterError)
from .image_utils import render_observation_images, render_evaluation_images
from .prompts import build_system_prompt, build_user_prompt
from .scoring import FIXED_PARAMS, PRIOR_CENTERS, inject_fixed_params
from .safe_eval import safe_evaluate

log = logging.getLogger(__name__)

ACTION_PATTERN = re.compile(
    r"<action>\s*tool:\s*(?P<tool>\w+)\s*(?P<body>.*?)</action>",
    re.DOTALL,
)

SOLUTION_PATTERN = re.compile(
    r"<solution_\d+>\s*(?P<json>\{.*?\})\s*</solution_\d+>",
    re.DOTALL,
)

LEGACY_INPUT_PATTERN = re.compile(
    r"input:\s*(?P<input>[\[{].*?[\]}])\s*$",
    re.DOTALL | re.MULTILINE,
)

MAX_CONTEXT_CHARS = 400_000


@dataclass
class AgentStep:
    step_num: int
    freeform_text: str
    tool_name: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    observation: str = ""
    observation_images: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


class InnerAgent:
    """ReAct agent that proposes gravitational lens model parameters.

    Each ``run()`` invocation consumes 5 context proposals from the
    database and returns one new proposal with its evaluation results.
    """

    def __init__(
        self,
        llm: OpenRouterClient,
        obs,
        *,
        max_steps: int = 15,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        eval_timeout_s: int = 60,
        system_prompt_override: Optional[str] = None,
        desc_llm: Optional[OpenRouterClient] = None,
        fixed_params: Optional[Dict[str, list]] = None,
        prior_centers: Optional[Dict[str, list]] = None,
        show_budget: bool = False,
        image_feedback_enabled: bool = True,
        finish_only_tool: bool = False,
        eval_results_postprocessor: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ):
        self.llm = llm
        self.obs = obs
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars
        self.eval_timeout_s = eval_timeout_s
        self._system_prompt_override = system_prompt_override
        self._desc_llm = desc_llm or llm
        self._fixed_params = fixed_params
        self._prior_centers = prior_centers
        self._show_budget = show_budget
        self._image_feedback_enabled = image_feedback_enabled
        self._finish_only_tool = finish_only_tool
        self._eval_results_postprocessor = eval_results_postprocessor

    def run(
        self,
        context_entries: List,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[AgentStep]]:
        """Execute the inner ReAct loop.

        Returns ``(proposal, eval_results, steps)``.  If the agent fails
        to produce a ``finish`` action within ``max_steps``, returns
        the best intermediate proposal (lowest image_chi2) or ``None``.
        """
        obs_images = (
            render_observation_images(self.obs)
            if self._image_feedback_enabled else
            {}
        )
        context_images = self._render_context_images(context_entries)

        messages = self._build_initial_messages(
            obs_images, context_entries, context_images)

        steps: List[AgentStep] = []
        best_proposal = None
        best_eval = None
        best_chi2 = float("inf")
        eval_count = 0

        for step_num in range(1, self.max_steps + 1):
            messages = self._maybe_trim_context(messages)

            log.debug("Inner agent step %d/%d  message_chars=%d",
                      step_num, self.max_steps,
                      sum(self._msg_char_len(m) for m in messages))

            t0 = time.time()
            try:
                content = self.llm.chat(messages)
            except BudgetExhausted:
                log.info("LLM budget exhausted at inner step %d", step_num)
                raise
            except ContextLengthExceeded:
                log.warning("Context length exceeded at step %d, trimming", step_num)
                messages = self._force_trim(messages)
                try:
                    content = self.llm.chat(messages)
                except ContextLengthExceeded:
                    log.error("Still exceeded after trim, aborting inner loop")
                    break
            except (OpenRouterError, Exception) as e:
                if isinstance(e, BudgetExhausted):
                    raise
                content = None
                for retry_i in range(4):
                    wait = 10 * (2 ** retry_i)
                    log.warning("LLM error at step %d (retry %d/4): %s. "
                                "Waiting %ds...",
                                step_num, retry_i + 1, str(e)[:100], wait)
                    time.sleep(wait)
                    try:
                        content = self.llm.chat(messages)
                        break
                    except BudgetExhausted:
                        raise
                    except Exception as e2:
                        e = e2
                if content is None:
                    log.warning("All retries failed at step %d. Continuing "
                                "to next step.", step_num)
                    continue
            elapsed = time.time() - t0

            freeform, tool_name, tool_input = self._parse_response(content)

            log.debug("Step %d  parsed: tool=%s  input_ok=%s  freeform_len=%d  "
                      "raw_tail=%.120s",
                      step_num, tool_name, tool_input is not None,
                      len(freeform),
                      content[-120:].replace('\n', '|'))

            step = AgentStep(
                step_num=step_num,
                freeform_text=freeform,
                tool_name=tool_name,
                tool_input=tool_input,
                elapsed_s=elapsed,
            )

            messages.append({"role": "assistant", "content": content})

            if tool_name == "evaluate" and self._finish_only_tool:
                correction = (
                    "The evaluate tool is disabled in this ablation. "
                    "Use only the finish tool and submit 3 final candidate "
                    "proposals inside <solution_1>, <solution_2>, <solution_3> tags."
                )
                step.observation = correction
                messages.append({"role": "user", "content": correction})
                steps.append(step)
                continue

            if tool_name in ("finish", "evaluate") and tool_input is not None:
                proposals = self._normalize_proposals(tool_input)
                if not proposals:
                    messages.append({
                        "role": "user",
                        "content": "Could not parse proposals. Provide a JSON "
                                   "array of 3 proposal objects.",
                    })
                    steps.append(step)
                    continue
                all_obs_text = []
                best_images = {}
                per_proposal_images: List[Dict[str, str]] = []

                for pi, raw_proposal in enumerate(proposals):
                    try:
                        proposal = self._inject_fixed_params(raw_proposal)
                    except ValueError as ve:
                        all_obs_text.append(
                            f"### Proposal {pi+1}: REJECTED — {ve}")
                        per_proposal_images.append({})
                        continue
                    eval_results, eval_err = self._run_evaluate(proposal)
                    if eval_results is None:
                        reason = eval_err or "timeout/unphysical"
                        if "list index out of range" in (eval_err or ""):
                            reason = (f"Wrong number of components. Check that "
                                      f"kwargs_lens has exactly "
                                      f"{len(self.obs.kwargs_model['lens_model_list'])} "
                                      f"items, kwargs_lens_light has "
                                      f"{len(self.obs.kwargs_model['lens_light_model_list'])}, "
                                      f"kwargs_source has "
                                      f"{len(self.obs.kwargs_model['source_light_model_list'])}.")
                        elif "unexpected keyword argument" in (eval_err or ""):
                            bad_kw = eval_err.split("'")[1] if "'" in eval_err else "?"
                            reason = (f"Invalid parameter '{bad_kw}'. "
                                      f"Check parameter names against the "
                                      f"component descriptions in the system prompt.")
                        all_obs_text.append(
                            f"### Proposal {pi+1}: FAILED — {reason}")
                        per_proposal_images.append({})
                        continue

                    from .scoring import residual_randomness as _rr
                    rr = _rr(eval_results)
                    if rr is not None:
                        eval_results["residual_randomness"] = rr

                    eval_count += 1
                    chi2 = eval_results.get("image_chi2_reduced", float("inf"))
                    prop_images = (
                        render_evaluation_images(self.obs, eval_results)
                        if self._image_feedback_enabled else
                        {}
                    )
                    per_proposal_images.append(prop_images)

                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_proposal = proposal
                        best_eval = eval_results
                        best_images = prop_images

                    obs_text = (
                        f"### Proposal {pi+1}\n"
                        + self._format_eval_observation(eval_results))
                    all_obs_text.append(obs_text)

                combined_text = "\n\n".join(all_obs_text)

                if tool_name == "finish":
                    step.observation = combined_text
                    steps.append(step)
                    return best_proposal, best_eval, steps

                descs = self._describe_all_proposals(per_proposal_images)
                if descs:
                    combined_text += "\n\n## Structure Analysis"
                    for pi, desc in descs.items():
                        combined_text += (
                            f"\n\n### Proposal {pi+1} — visual analysis\n"
                            + desc)

                if self._show_budget:
                    remaining = self.max_steps - step_num
                    combined_text += (
                        f"\n\n---\nStep {step_num}/{self.max_steps} "
                        f"({remaining} remaining). "
                        f"Best chi2 so far: {best_chi2:.6f}. "
                        f"Call **finish** when satisfied or to submit your best before steps run out.")

                step.observation = combined_text
                step.observation_images = (
                    list(best_images.values()) if self._image_feedback_enabled else []
                )
                obs_msg = self._build_observation_message(
                    combined_text, best_images)
                messages.append(obs_msg)

                log.debug("  eval %d proposals, best chi2=%.6f",
                          len(proposals), best_chi2)
            elif tool_name in ("evaluate", "finish") and tool_input is None:
                allowed_tool = "finish" if self._finish_only_tool else tool_name
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your {allowed_tool} action could not be parsed. The JSON "
                        "input must be valid and on a SINGLE line or properly "
                        "formatted. Here is the exact format:\n\n"
                        '<action>\n'
                        f'tool: {allowed_tool}\n'
                        'input: {"kwargs_lens": [{"k_eff": 0.2, "R_sersic": 1.0, '
                        '"n_sersic": 3.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, '
                        '"center_y": 0.0}, {"Rs": 2.0, "alpha_Rs": 1.0, '
                        '"fR0": 5e-7, "center_x": 0.0, "center_y": 0.0}, '
                        '{"gamma1": 0.0, "gamma2": 0.0}], '
                        '"kwargs_lens_light": [{"n_sersic": 3.0, "R_sersic": 0.8, '
                        '"e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}], '
                        '"kwargs_source": [{"n_sersic": 1.5, "R_sersic": 0.3, '
                        '"e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}]}\n'
                        '</action>\n\n'
                        'Try again with valid JSON.'
                    ),
                })
            elif tool_name is None:
                required_tool = "finish" if self._finish_only_tool else "evaluate"
                messages.append({
                    "role": "user",
                    "content": (
                        "No <action> block detected in your response. You MUST "
                        f"call the {required_tool} tool now. Write an <action> block with "
                        f"tool: {required_tool} and the input JSON. Example:\n\n"
                        '<action>\n'
                        f'tool: {required_tool}\n'
                        'input: {"kwargs_lens": [...], "kwargs_lens_light": [...], '
                        '"kwargs_source": [...]}\n'
                        '</action>'
                    ),
                })
            else:
                allowed_tools = "finish only" if self._finish_only_tool else "evaluate or finish"
                messages.append({
                    "role": "user",
                    "content": f"Unknown tool '{tool_name}'. Use {allowed_tools}.",
                })

            steps.append(step)

        log.debug("  max_steps reached, best intermediate chi2=%.6f", best_chi2)
        return best_proposal, best_eval, steps

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, text: str,
    ) -> Tuple[str, Optional[str], Optional[Any]]:
        """Extract freeform text and the last <action> block.

        Supports two formats:
          1. <solution_1>...<solution_3> tags inside <action> (preferred)
          2. Legacy: input: {single JSON} inside <action> (fallback)

        Returns (freeform_text, tool_name, parsed_input).
        parsed_input is a list of dicts (solution tags) or a single dict.
        """
        match = None
        for m in ACTION_PATTERN.finditer(text):
            match = m

        if match is None:
            return text.strip(), None, None

        freeform = text[:match.start()].strip()
        tool_name = match.group("tool").strip().lower()
        body = match.group("body")

        solutions = []
        for sm in SOLUTION_PATTERN.finditer(body):
            try:
                solutions.append(json.loads(sm.group("json")))
            except json.JSONDecodeError:
                log.debug("Failed to parse solution JSON: %s",
                          sm.group("json")[:100])

        if solutions:
            return freeform, tool_name, solutions

        legacy = LEGACY_INPUT_PATTERN.search(body)
        if legacy:
            raw = legacy.group("input").strip()
            try:
                parsed = json.loads(raw)
                return freeform, tool_name, parsed
            except json.JSONDecodeError:
                log.debug("Failed to parse legacy input: %s", raw[:100])

        log.debug("No parseable input in action block")
        return freeform, tool_name, None

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _run_evaluate(self, proposal: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Call the pipeline's evaluate_proposal with timeout protection.

        Returns (result_dict, error_string).
        """
        from . import scoring as _S
        result, err = safe_evaluate(
            proposal, self.obs,
            include_kinematics=True,
            subtracted_chi2=_S.SUBTRACTED_CHI2,
            no_linear_solve=_S.NO_LINEAR_SOLVE,
            timeout_s=self.eval_timeout_s,
        )
        if result is not None and self._eval_results_postprocessor is not None:
            result = self._eval_results_postprocessor(proposal, result)
        if err:
            log.warning("Evaluation failed: %s", err)
        return result, err

    _DESC_SYSTEM = (
        "You are an expert at analyzing gravitational lens images. You will "
        "receive a multi-panel comparison image. The panels are:\n"
        "1) GT (Observed) — the raw data (foreground galaxy + lensed arcs)\n"
        "2) GT-Lens — observed data minus model lens light, revealing actual lensed arcs/rings\n"
        "3) Render — full model (lens light + lensed source)\n"
        "4+) Residual and/or Render_Sub (model's predicted arcs) and Residual_Sub\n\n"
        "In residual panels: BLUE = model too bright, RED = model too faint, "
        "WHITE = good fit.\n\n"
        "Your job: describe the STRUCTURE of the lensed arcs visible in the "
        "GT-Lens panel (panel 2), and what the residual shows the model is missing.\n"
        "- What arc/ring geometry is visible in GT-Lens? "
        "Partial ring, full Einstein ring, multiple arcs, asymmetric arcs?\n"
        "- What is the angular extent and thickness of the arcs?\n"
        "- Compare GT-Lens (real arcs) with Render_Sub (model arcs) — what is missing?\n"
        "- What structures appear in the residual? Rings, arcs, dipoles, "
        "gradients?\n\n"
        "Be concrete, quantitative, and brief."
    )

    def _describe_all_proposals(
        self, per_proposal_images: List[Dict[str, str]],
    ) -> Dict[int, str]:
        """Call the LLM once per proposal with the 5-panel image set."""
        if not self._image_feedback_enabled:
            return {}
        results: Dict[int, str] = {}
        for pi, images in enumerate(per_proposal_images):
            if not images.get("comparison"):
                continue
            desc = self._describe_single_proposal(images)
            if desc:
                results[pi] = desc
        return results

    def _describe_single_proposal(
        self, images: Dict[str, str],
    ) -> Optional[str]:
        """One LLM call with the comparison strip."""
        from .scoring import SUBTRACTED_CHI2
        comparison = images.get("comparison")
        if not comparison:
            return None
        if SUBTRACTED_CHI2:
            panel_desc = (
                "5-panel comparison image. Left to right:\n"
                "1) GT (Observed) — the real data\n"
                "2) GT - Lens (Arcs Only) — observed DATA minus model lens light, revealing actual lensed arcs/rings in the data\n"
                "3) Render — full model (lens light + lensed source)\n"
                "4) Render_Sub (Model Arcs) — full MODEL minus model lens light, showing what the model predicts for the lensed source only\n"
                "5) Residual_Sub — residual of source-only fit after lens light removal, blue=too bright, red=too faint")
            analysis_prompt = (
                "Analyze the 5 panels. Focus on:\n"
                "- What arc/ring structure is visible in GT - Lens (panel 2)? These are the REAL arcs in the data.\n"
                "- Compare GT - Lens (panel 2) with Render_Sub (panel 4) — panel 2 is the real arcs, panel 4 is the model's prediction. What structures are not captured?\n"
                "- What does the Residual_Sub (panel 5) show the model is missing?\n"
                "- Describe morphology, positions, and extents of features.")
        else:
            panel_desc = (
                "6-panel comparison image. Left to right:\n"
                "1) GT (Observed) — the real data\n"
                "2) GT - Lens (Arcs Only) — observed DATA minus model lens light, revealing actual lensed arcs/rings in the data\n"
                "3) Render — full model (lens light + lensed source)\n"
                "4) Residual — (model-data)/noise, blue=too bright, red=too faint\n"
                "5) Render_Sub (Model Arcs) — full MODEL minus model lens light, showing what the model predicts for the lensed source only\n"
                "6) Residual_Sub — residual of source-only fit after lens light removal")
            analysis_prompt = (
                "Analyze the 6 panels. Focus on:\n"
                "- What arc/ring structure is visible in GT - Lens (panel 2)? These are the REAL arcs in the data.\n"
                "- Compare GT - Lens (panel 2) with Render_Sub (panel 5) — panel 2 is the real arcs, panel 5 is the model's prediction. What structures are not captured?\n"
                "- What does the Residual (panel 4) show the model is missing?\n"
                "- Does Residual_Sub (panel 6) show remaining source structure?\n"
                "- Describe morphology, positions, and extents of features.")
        content_parts: list = [
            {"type": "text", "text": panel_desc},
            {"type": "image_url", "image_url": {"url": comparison}},
            {"type": "text", "text": analysis_prompt},
        ]
        try:
            desc = self._desc_llm.chat([
                {"role": "system", "content": self._DESC_SYSTEM},
                {"role": "user", "content": content_parts},
            ], count_toward_budget=False)
            if desc and desc.strip():
                log.debug("Proposal description: %s", desc.strip()[:150])
                return desc.strip()
            return None
        except Exception as e:
            log.warning("Proposal description failed: %s", e)
            return None

    def _normalize_proposals(
        self, tool_input: Any,
    ) -> List[Dict[str, Any]]:
        """Accept either a single proposal dict or a list of up to 3."""
        if isinstance(tool_input, list):
            return [p for p in tool_input
                    if isinstance(p, dict) and 'kwargs_lens' in p][:3]
        if isinstance(tool_input, dict) and 'kwargs_lens' in tool_input:
            return [tool_input]
        return []

    def _inject_fixed_params(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Merge fixed params and pad missing components.

        Raises ValueError if the LLM provides too many components so the
        error propagates back to the agent as actionable feedback.
        """
        return inject_fixed_params(
            proposal,
            fixed_params=self._fixed_params if self._fixed_params is not None else FIXED_PARAMS,
            prior_centers=self._prior_centers if self._prior_centers is not None else PRIOR_CENTERS,
        )

    def _format_eval_observation(self, eval_results: Dict[str, Any]) -> str:
        """Format evaluation results as text for the agent context."""
        from .scoring import BLIND_MODE
        chi2_img = eval_results.get('image_chi2_reduced')

        lines = ["## Evaluation Results"]

        if chi2_img is not None:
            target_chi2 = 1.0
            lines.append(f"Image chi2_reduced: {chi2_img:.6f}  (target: ~{target_chi2:.1f}, gap: {chi2_img - target_chi2:+.6f})")
        else:
            lines.append("Image chi2_reduced: N/A")

        masses = eval_results.get("masses_msun")
        mass_cap = eval_results.get("subhalo_mass_cap_msun")
        if masses:
            mass_text = ", ".join(f"{mass:.2e}" for mass in masses)
            lines.append(f"Subhalo masses (M200): {mass_text} Msun")
        if mass_cap is not None:
            if eval_results.get("subhalo_mass_limit_ok", True):
                lines.append(f"Subhalo mass cap: OK  (all <= {mass_cap:.2e} Msun)")
            else:
                max_mass = eval_results.get("subhalo_mass_max_msun", 0.0)
                lines.append(f"Subhalo mass cap: VIOLATION  (max={max_mass:.2e} > {mass_cap:.2e} Msun)")
                reason = eval_results.get("subhalo_mass_violation_reason")
                if reason:
                    lines.append(f"  WARNING: {reason} This proposal is penalized and excluded from final ranking.")

        if not BLIND_MODE:
            sigma_pred = eval_results.get('sigma_predicted')
            sigma_obs = eval_results.get('sigma_observed', self.obs.sigma_obs)
            sigma_err = eval_results.get('sigma_observed_err', self.obs.sigma_obs_err)
            kin_chi2 = eval_results.get('kin_chi2')

            from .scoring import KIN_SOFT
            if sigma_pred is not None and sigma_pred > 0 and sigma_obs:
                delta = sigma_pred - sigma_obs
                within = abs(delta) <= sigma_err if sigma_err else False
                if KIN_SOFT and within:
                    lines.append(f"Velocity dispersion: predicted={sigma_pred:.1f}  observed={sigma_obs:.1f} +/- {sigma_err:.1f}  — WITHIN ERROR, score=0 (no penalty)")
                else:
                    n_sig = abs(delta) / sigma_err if sigma_err else 0
                    lines.append(f"Velocity dispersion: predicted={sigma_pred:.1f}  observed={sigma_obs:.1f} +/- {sigma_err:.1f}  (delta={delta:+.1f}, {n_sig:.1f} sigma off)")
            elif sigma_obs:
                lines.append(f"Velocity dispersion: FAILED (observed={sigma_obs:.1f} +/- {sigma_err:.1f})")

            if kin_chi2 is not None:
                if KIN_SOFT and kin_chi2 <= 1.0:
                    lines.append(f"Kinematic chi2: {kin_chi2:.3f}  — within error, no penalty")
                else:
                    lines.append(f"Kinematic chi2: {kin_chi2:.3f}")

            rand = eval_results.get("residual_randomness")
            if rand is not None:
                lines.append(f"Residual randomness: {rand:.4f}  (lower=more random=better, target: <0.1)")

        from .scoring import PHYSICALITY_MODE
        if PHYSICALITY_MODE in ("active", "post"):
            rmse_p = eval_results.get("rmse_poisson")
            is_phys = eval_results.get("is_physical")
            min_k = eval_results.get("min_kappa")
            neg_frac = eval_results.get("negative_mass_frac")
            if rmse_p is not None:
                status = "PHYSICAL" if is_phys else "UNPHYSICAL"
                lines.append(f"Physicality: {status}  Poisson RMSE={rmse_p:.4f} (threshold: 0.05)  "
                             f"min_kappa={min_k:.3f}  negative_mass_frac={neg_frac:.3f}")
                if not is_phys:
                    lines.append("  WARNING: This proposal is UNPHYSICAL and will be excluded from final ranking. "
                                 "Adjust mass parameters to produce a smoother, more physical convergence field.")

        lines.append("")
        lines.append(
            "Images attached: observed vs model vs residual."
            if self._image_feedback_enabled else
            "Images are disabled in this ablation; rely on the numeric diagnostics above."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self,
        obs_images: Dict[str, str],
        context_entries: List,
        context_images: Dict[int, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Build the initial system + user messages with optional images."""
        content_blocks: List[Dict[str, Any]] = []

        if self._image_feedback_enabled:
            content_blocks.append({
                "type": "text",
                "text": "## Observed Image\n",
            })
            if "observed" in obs_images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": obs_images["observed"]},
                })

        if self._image_feedback_enabled:
            for i, entry in enumerate(context_entries):
                imgs = context_images.get(i, {})
                content_blocks.append({
                    "type": "text",
                    "text": f"\n## Reference Proposal {i+1} Images\n",
                })
                if "comparison" in imgs:
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": imgs["comparison"]},
                    })

        user_text = build_user_prompt(
            context_entries,
            obs_sigma=self.obs.sigma_obs,
            obs_sigma_err=self.obs.sigma_obs_err,
            z_lens=self.obs.z_lens,
            z_source=self.obs.z_source,
            kwargs_model=getattr(self.obs, 'kwargs_model', None),
            image_feedback_enabled=self._image_feedback_enabled,
        )
        content_blocks.append({"type": "text", "text": user_text})

        sys_prompt = self._system_prompt_override or build_system_prompt(
            available_tools=["finish"] if self._finish_only_tool else ["evaluate", "finish"],
            image_feedback_enabled=self._image_feedback_enabled,
        )
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content_blocks},
        ]

    def _build_observation_message(
        self, text: str, images: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build a user message with evaluation text and optional images."""
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        if self._image_feedback_enabled:
            for label in ("comparison", "residual"):
                if label in images:
                    blocks.append({"type": "text", "text": f"\n[{label}]"})
                    blocks.append({
                        "type": "image_url",
                        "image_url": {"url": images[label]},
                    })
        return {"role": "user", "content": blocks}

    def _render_context_images(
        self, entries: List,
    ) -> Dict[int, Dict[str, str]]:
        """Pre-render comparison images for each context entry."""
        if not self._image_feedback_enabled:
            return {}
        result = {}
        for i, entry in enumerate(entries):
            er = entry.eval_results
            if "model_image" not in er:
                er, _ = self._run_evaluate(entry.proposal)
                if er is None:
                    continue
            result[i] = render_evaluation_images(self.obs, er)
        return result

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_char_len(msg: Dict[str, Any]) -> int:
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if block.get("type") == "text":
                    total += len(block.get("text", ""))
                elif block.get("type") == "image_url":
                    total += 200
            return total
        return 0

    def _maybe_trim_context(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop middle conversation turns if context exceeds the cap."""
        total = sum(self._msg_char_len(m) for m in messages)
        if total <= self.max_context_chars:
            return messages

        keep_start = 2
        keep_end = 10
        if len(messages) <= keep_start + keep_end:
            return messages

        trimmed = messages[:keep_start] + messages[-keep_end:]
        log.info("Trimmed context from %d to %d messages "
                 "(%d -> %d chars)",
                 len(messages), len(trimmed), total,
                 sum(self._msg_char_len(m) for m in trimmed))
        return trimmed

    def _force_trim(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggressive trim: keep system + initial user + last 4 turns."""
        keep_start = 2
        keep_end = 4
        if len(messages) <= keep_start + keep_end:
            return messages
        return messages[:keep_start] + messages[-keep_end:]
