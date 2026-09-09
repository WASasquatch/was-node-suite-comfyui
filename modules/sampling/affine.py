"""Injecting an affine into a running sampler.

:func:`patch_sampler` wraps a SAMPLER's ``sampler_function`` and transforms the sampler's
own ``x`` in place at the opening model call of each scheduled step.
"""

from __future__ import annotations

import torch

from .. import log
from ..latent import affine
from ..latent import affine_patterns as patterns

__all__ = [
    "ACTS_ON",
    "UNHOOKABLE",
    "AffineSpec",
    "patch_sampler",
]

logger = log.get_logger("sampling.affine")

#: What the multiplier is applied to at a scheduled step.
ACTS_ON = ["content", "latent"]

#: Sampler functions whose model call does not carry the sampler's own state, keyed by the
#: ``__name__`` a KSAMPLER wraps.
UNHOOKABLE = {
    "dpm_fast_function": (
        "dpm_fast drives an internal solver with sigmas of its own, so no step boundary "
        "can be found"
    ),
    "dpm_adaptive_function": (
        "dpm_adaptive drives an internal solver with sigmas of its own, so no step boundary "
        "can be found"
    ),
    "sample_unipc": (
        "uni_pc rescales the latent before every model call, so a transform there does not "
        "reach the trajectory"
    ),
    "sample_unipc_bh2": (
        "uni_pc_bh2 rescales the latent before every model call, so a transform there does "
        "not reach the trajectory"
    ),
    "sample_ar_video": (
        "the autoregressive video sampler builds a fresh input per window rather than "
        "stepping one latent"
    ),
}


class AffineSpec:
    """Everything an inline affine needs, resolved once when the node runs.

    Attributes:
        schedule: The per-step strength curve, as :func:`modules.latent.affine.step_schedule`
            reads it.
        interval: Apply on every Nth step.
        max_scale: Multiplier reached at the peak of the schedule.
        max_bias: Offset reached at the peak of the schedule.
        pattern: Mask pattern name.
        pattern_asked: The pattern the sampler's own widget named, where an Affine Options
            node named a different one.
        seed: Seeds the mask.
        seed_increment: Whether the seed advances on every application.
        temporal_mode: ``"static"``, ``"per_frame"`` or ``"drift"``.
        streams: ``"video"``, ``"audio"`` or ``"both"``.
        external_mask: A supplied MASK, or None.
        options: Pattern parameters and mask shaping.
        space: ``"latent"`` to read the values against a node-space latent, ``"model"`` to
            apply them to the sampler's own latent.
        acts_on: ``"content"`` to multiply only the part of the latent that is picture,
            ``"latent"`` to multiply the whole latent, noise included.
        total_steps: Steps the schedule spans, where the run covers only part of it.
        step_offset: Where in that schedule the run starts.
        debug: Whether every application is logged.
    """

    def __init__(
        self,
        schedule=None,
        interval=1,
        max_scale=1.02,
        max_bias=0.0,
        pattern="white_noise",
        seed=0,
        seed_increment=False,
        temporal_mode="static",
        streams="video",
        external_mask=None,
        options=None,
        space="latent",
        acts_on="content",
        total_steps=None,
        step_offset=0,
        debug=False,
        pattern_asked=None,
    ):
        self.pattern_asked = None if pattern_asked is None else str(pattern_asked)
        self.schedule = dict(schedule) if isinstance(schedule, dict) else dict(affine.SCHEDULE_DEFAULTS)
        self.interval = max(1, int(interval))
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.pattern = str(pattern)
        self.seed = int(seed)
        self.seed_increment = bool(seed_increment)
        self.temporal_mode = str(temporal_mode)
        self.streams = str(streams)
        self.external_mask = external_mask
        self.options = affine.resolve(options)
        self.space = str(space or "latent")
        self.acts_on = str(acts_on or "content")
        self.total_steps = None if total_steps is None else int(total_steps)
        self.step_offset = int(step_offset)
        self.debug = bool(debug)

    @property
    def is_noop(self) -> bool:
        """Whether the settings leave the latent exactly as it was."""
        return abs(self.max_scale - 1.0) < 1e-8 and abs(self.max_bias) < 1e-8

    def inert(self) -> list[str]:
        """Settings that are on but cannot reach the result as the rest are set.

        Returns:
            One sentence per setting, each naming what to change. Empty where every
            setting in play does something.
        """
        from ..latent import affine_patterns as patterns

        out = []
        if self.pattern_asked and self.pattern_asked != self.pattern:
            out.append(
                f"pattern: sampler '{self.pattern_asked}', affine_options "
                f"'{self.pattern}', affine_options wins"
            )
        if str(self.options.get("bias_field", "constant")) == "gaussian" and abs(self.max_bias) < 1e-8:
            out.append(
                "bias_field 'gaussian', max_bias 0.0: no field applied"
            )
        if self.pattern in patterns.CONTENT_PATTERNS and self.temporal_mode != "static":
            out.append(
                f"temporal_mode '{self.temporal_mode}' not read: '{self.pattern}' is "
                "read per frame"
            )
        if str(self.options.get("content_gate", "off")) != "off" and self.pattern in patterns.CONTENT_PATTERNS:
            out.append(
                f"content_gate not read: '{self.pattern}' already reads the picture"
            )
        return out

    def multipliers(self, steps: int) -> list[float]:
        """The strength curve over the steps this run covers.

        Args:
            steps: How many steps the run has.

        Returns:
            One value per step, each 0.0 to 1.0.
        """
        total = max(int(self.total_steps or steps), 1)
        curve = affine.step_schedule(total, self.schedule)
        out = []
        for i in range(steps):
            at = i + self.step_offset
            out.append(float(curve[at]) if 0 <= at < len(curve) else 0.0)
        return out


def _walk(model, attribute: str, depth: int = 4):
    """Find the first model in an inner-model chain carrying an attribute.

    Args:
        model: The callable a sampler was handed.
        attribute: The attribute name to look for.
        depth: How many links to follow.

    Returns:
        The node carrying it, or None.
    """
    node = model
    for _ in range(depth):
        if node is None:
            return None
        if getattr(node, attribute, None) is not None:
            return node
        node = getattr(node, "inner_model", None)
    return None


def _latent_shapes(model, x: torch.Tensor) -> list:
    """The shapes a packed multi-stream latent splits back into.

    Args:
        model: The callable a sampler was handed.
        x: The packed latent.

    Returns:
        One shape per stream, or the packed tensor's own shape where nothing recorded them.
    """
    holder = _walk(model, "latent_shapes")
    shapes = getattr(holder, "latent_shapes", None) if holder is not None else None
    return list(shapes) if shapes else [tuple(x.shape)]


def _uniform(t: torch.Tensor):
    """Collapse a probe result to a number where it holds one value.

    Args:
        t: A tensor.

    Returns:
        A float where every entry matches, the tensor otherwise.
    """
    try:
        low = float(t.min())
        high = float(t.max())
        if abs(high - low) <= max(1e-9, abs(high) * 1e-6):
            return 0.5 * (low + high)
    except Exception:
        pass
    return t


def _probe_space(model, x: torch.Tensor, shapes: list):
    """Measure the ``slope * z + offset`` a model applies on the way into sampling.

    Args:
        model: The callable a sampler was handed.
        x: The packed latent.
        shapes: One shape per stream.

    Returns:
        A ``(slope, offset)`` pair per stream, or None where the probe could not run.
    """
    import comfy.utils

    base = _walk(model, "process_latent_in")
    if base is None:
        return None
    try:
        if len(shapes) > 1:
            offset = base.process_latent_in(torch.zeros_like(x))
            slope = base.process_latent_in(torch.ones_like(x)) - offset
            offsets = comfy.utils.unpack_latents(offset, shapes)
            slopes = comfy.utils.unpack_latents(slope, shapes)
        else:
            shape = tuple(shapes[0])
            probe = (1, shape[1]) + (1,) * (len(shape) - 2)
            zeros = torch.zeros(probe, device=x.device, dtype=x.dtype)
            ones = torch.ones(probe, device=x.device, dtype=x.dtype)
            offset = base.process_latent_in(zeros)
            slopes, offsets = [base.process_latent_in(ones) - offset], [offset]
        return [(_uniform(s), _uniform(o)) for s, o in zip(slopes, offsets)]
    except Exception:
        return None


def _signal_coefficient(model, sigma: float, like: torch.Tensor):
    """What fraction of a clean latent survives into the sampler's latent at one noise level.

    Args:
        model: The callable a sampler was handed.
        sigma: The noise level of the step.
        like: A tensor whose device and dtype the probe follows.

    Returns:
        The coefficient as a float, or None where the model could not be asked. Flow models
        answer ``1 - sigma`` and eps models answer ``1``.
    """
    holder = _walk(model, "model_sampling")
    sampling = getattr(holder, "model_sampling", None) if holder is not None else None
    if sampling is None:
        return None
    try:
        level = torch.tensor([float(sigma)], device=like.device, dtype=torch.float32)
        zeros = torch.zeros((1, 1), device=like.device, dtype=torch.float32)
        ones = torch.ones((1, 1), device=like.device, dtype=torch.float32)
        return float(sampling.noise_scaling(level, zeros, ones).flatten()[0])
    except Exception:
        return None


def _to_model_space(scale, bias, slope, offset):
    """Restate a node-space affine in the space the sampler carries.

    Args:
        scale: Node-space multiplier.
        bias: Node-space offset.
        slope: What the model multiplies a node-space latent by.
        offset: What the model adds to it.

    Returns:
        ``(scale, bias, bias_dc)`` in model space. The multiplier is unchanged, and the
        third value is the part belonging to the multiplier rather than to the offset.
    """
    return scale, slope * bias, offset * (1.0 - scale)


class _ModelProxy:
    """A stand-in for the sampler's model that reports every call and passes the rest through."""

    def __init__(self, inner, on_call):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_on_call", on_call)

    def __call__(self, x, sigma, *args, **kwargs):
        object.__getattribute__(self, "_on_call")(x, sigma)
        return object.__getattribute__(self, "_inner")(x, sigma, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)


class _Run:
    """One sampling run: which steps to act on, where they are, and the transform itself."""

    def __init__(self, spec: AffineSpec, model, x: torch.Tensor, sigmas: torch.Tensor, holder: dict):
        self.spec = spec
        self.model = model
        self.holder = holder
        self.x = x
        self.sigma_values = [float(s) for s in sigmas.flatten().tolist()]
        self.steps = max(int(sigmas.shape[-1]) - 1, 0)

        self.shapes = _latent_shapes(model, x)
        self.packed = len(self.shapes) > 1
        self.stream_ids = affine.stream_indices(spec.streams, len(self.shapes))
        self.space = None
        self.probed = False

        self.multipliers = spec.multipliers(self.steps)
        self.targets = self._targets()

        self.calls = 0
        self.callbacks = 0
        self.denoised = None
        self.matches = 0
        self.occurrence = None
        self.applied = set()
        self.applications = 0
        self.landed = []
        self.uncoupled = False
        self.share = 1.0
        self.gain = 1.0
        self.reached = []
        self.guessed = False
        self.failed = set()
        self.degraded = set()
        self.waited = False

    def _targets(self) -> set:
        """Which local steps the affine lands on.

        Returns:
            Step indices, empty where the settings do nothing.
        """
        if self.spec.is_noop or not self.stream_ids:
            return set()
        out = set()
        for i in range(self.steps):
            if ((i + self.spec.step_offset + 1) % self.spec.interval) != 0:
                continue
            if self.multipliers[i] > 1e-8:
                out.add(i)
        return out

    def proxy(self, model):
        """Wrap the sampler's model so every call is seen.

        Args:
            model: The callable the sampler was handed.

        Returns:
            A proxy standing in for it.
        """
        return _ModelProxy(model, self.on_model_call)

    def callback(self, user_callback):
        """Wrap the sampler's callback so every step boundary is seen.

        Args:
            user_callback: The callback the sampler was handed, or None.

        Returns:
            A callback to hand the sampler instead.
        """

        def wrapped(d):
            # The model's estimate of the clean latent.
            estimate = d.get("denoised") if isinstance(d, dict) else None
            if torch.is_tensor(estimate):
                self.denoised = estimate
            self.on_callback()
            if user_callback is not None:
                return user_callback(d)
            return None

        return wrapped

    def on_callback(self) -> None:
        """Close a step and learn how many calls at its own sigma the sampler makes."""
        step = self.callbacks
        if step >= 1 and self.matches >= 1:
            learned = self.matches
            if self.occurrence is None and learned > 1 and self.guessed:
                logger.warning(
                    "The affine on step 1 may have landed on a substep: this sampler "
                    "evaluates the model %s times at each step's own noise level, and the "
                    "first of those belongs to the step before. Every later step is exact.",
                    learned,
                )
            self.occurrence = learned
        if step in self.targets and step not in self.applied and self.spec.debug:
            logger.info("Affine step %s was scheduled but no model call matched it.", step)
        self.callbacks += 1
        self.matches = 0

    def on_model_call(self, x, sigma) -> None:
        """Transform the latent where this call opens a scheduled step.

        Args:
            x: The tensor the sampler is about to denoise.
            sigma: The noise level it is denoising at.
        """
        self.calls += 1
        step = self.callbacks
        if step >= self.steps or not self._at_step_sigma(step, sigma):
            return
        self.matches += 1
        if step not in self.targets or step in self.applied:
            return
        # Most samplers reach a step's noise level once, the heun family twice.
        wanted = 1 if (step == 0 or self.occurrence is None) else self.occurrence
        if self.matches != wanted:
            return
        if self.occurrence is None and step > 0:
            self.guessed = True
        self.applied.add(step)
        self.apply(x, step)

    def _at_step_sigma(self, step: int, sigma) -> bool:
        """Whether a model call is being made at one step's own noise level.

        Args:
            step: The step index being tested.
            sigma: The noise level of the call.

        Returns:
            True where the two match.
        """
        try:
            value = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
        except Exception:
            return False
        target = self.sigma_values[step]
        return abs(value - target) <= max(1e-6, abs(target) * 1e-4)

    def _space_for(self, index: int, scale: float, bias: float):
        """The scale and bias to use on one stream of the sampler's own latent.

        Args:
            index: Stream index.
            scale: Node-space multiplier.
            bias: Node-space offset.

        Returns:
            ``(scale, bias, bias_dc)`` as the stream wants them.
        """
        if self.spec.space != "latent":
            return scale, bias, 0.0
        if not self.probed:
            self.probed = True
            self.space = _probe_space(self.model, self.x, self.shapes)
            if self.space is None and self.spec.debug:
                logger.info("The latent-space probe failed; applying the raw values instead.")
        if not self.space or index >= len(self.space):
            return scale, bias, 0.0
        slope, offset = self.space[index]
        return _to_model_space(scale, bias, slope, offset)

    def apply(self, x: torch.Tensor, step: int) -> None:
        """Transform the sampler's latent in place for one step.

        Args:
            x: The tensor the sampler is about to denoise.
            step: The step index the strength is read at.
        """
        transformed = self.transform(x, step, self._clean_streams())
        if transformed is None:
            return
        x.copy_(transformed)

    def release(self) -> None:
        """Drop the tensors and the model the run held, keeping only its counts."""
        self.x = None
        self.denoised = None
        self.model = None
        self.space = None

    def transform(self, x: torch.Tensor, step: int, clean: list):
        """The affine over every selected stream of one tensor.

        Args:
            x: The tensor to transform.
            step: The step index the strength is read at.
            clean: One clean estimate per stream, for a content-aware pattern.

        Returns:
            A new tensor, or None where no stream was transformed.
        """
        import comfy.utils

        spec = self.spec
        strength = self.multipliers[step]
        scale = 1.0 + (spec.max_scale - 1.0) * strength
        bias = spec.max_bias * strength
        seed = spec.seed + (self.applications if spec.seed_increment else 0)

        parts = comfy.utils.unpack_latents(x, self.shapes) if self.packed else [x]
        out = list(parts)
        touched = False
        for index in self.stream_ids:
            if index >= len(parts):
                continue
            source = parts[index]
            if not torch.is_tensor(source) or source.ndim not in (3, 4, 5):
                continue
            content = clean[index] if index < len(clean) else None
            picture = self._picture(source, content, step)
            if picture is None and spec.acts_on == "content":
                continue
            transformed, mask = self._one_stream(
                source, index, scale, bias, seed, content, picture
            )
            if transformed is None:
                continue
            out[index] = transformed
            touched = True
            if index == self.stream_ids[0]:
                weight = float(mask.float().mean()) if torch.is_tensor(mask) else 1.0
                self.gain *= 1.0 + (scale - 1.0) * weight
                self.reached.append(self.share if spec.acts_on == "content" else 1.0)
            if self.holder.get("mask") is None or index == self.stream_ids[0]:
                self.holder["mask"] = mask

        if not touched:
            return None
        self.applications += 1
        self.landed.append((step, self.sigma_values[step]))
        if spec.debug:
            logger.info(
                "Affine step %s on the %s: sigma %.4f, strength %.3f, scale %.4f, "
                "bias %.4f, seed %s, streams %s",
                step,
                spec.acts_on,
                self.sigma_values[step],
                strength,
                scale,
                bias,
                seed,
                self.stream_ids,
            )
        if self.packed:
            packed, _ = comfy.utils.pack_latents(out)
            return packed
        return out[0]

    def _picture(self, source: torch.Tensor, content, step: int):
        """The part of one stream that is picture rather than noise.

        Args:
            source: The stream's tensor.
            content: The model's clean estimate for it, or None.
            step: The step index, which sets the noise level.

        Returns:
            The scaled clean estimate, or None where the whole latent is to be multiplied
            or the estimate is not there yet.
        """
        if self.spec.acts_on != "content":
            return None
        if content is None:
            if not self.waited:
                self.waited = True
                logger.warning(
                    "affine_acts_on is 'content', which multiplies the picture the model "
                    "has resolved so far, and no step has finished yet, so this step was "
                    "left alone. Start the schedule a step or two later."
                )
            return None
        share = _signal_coefficient(self.model, self.sigma_values[step], source)
        self.share = 1.0 if share is None else float(share)
        if share is None:
            if not self.uncoupled:
                self.uncoupled = True
                logger.warning(
                    "The model would not say how much of a clean latent survives at this "
                    "noise level, so nothing was applied. Set affine_acts_on to 'latent' "
                    "to multiply the whole latent instead."
                )
            return None
        return content * share

    def _clean_streams(self) -> list:
        """The model's clean estimate, split the way the latent is.

        Returns:
            One tensor per stream, or an empty list before the first estimate arrives.
        """
        import comfy.utils

        if not torch.is_tensor(self.denoised):
            return []
        try:
            if self.packed:
                return comfy.utils.unpack_latents(self.denoised, self.shapes)
            return [self.denoised]
        except Exception:
            return []

    def _one_stream(
        self,
        source: torch.Tensor,
        index: int,
        scale: float,
        bias: float,
        seed: int,
        content=None,
        picture=None,
    ):
        """Transform one stream, reporting a failure rather than letting it stop the run.

        Args:
            source: The stream's tensor.
            index: Stream index.
            scale: Node-space multiplier.
            bias: Node-space offset.
            seed: Seeds the mask.
            content: The model's clean estimate for this stream, for a content-aware pattern.
            picture: The part of the stream that is picture, where the multiplier is to be
                applied to that alone.

        Returns:
            ``(latent, mask)``, or ``(None, None)`` where the transform raised.
        """
        spec = self.spec
        pattern = spec.pattern
        if pattern in patterns.CONTENT_PATTERNS and content is None:
            if not self.waited:
                self.waited = True
                logger.warning(
                    "'%s' reads the picture the model has resolved so far, and no step has "
                    "finished yet, so this one was left alone. Start the schedule a step or "
                    "two later to give it something to read.",
                    pattern,
                )
            return None, None
        if pattern in patterns.CONTENT_PATTERNS and not affine.content_pattern_fits(source):
            if index not in self.degraded:
                self.degraded.add(index)
                logger.info(
                    "Stream %s is too small for '%s', so 'solid' was used on it instead.",
                    index,
                    pattern,
                )
            pattern = "solid"
        value, offset, offset_dc = self._space_for(index, scale, bias)
        try:
            return affine.apply_plane(
                source,
                value,
                offset,
                pattern,
                spec.temporal_mode,
                seed + index,
                spec.external_mask,
                spec.options,
                content,
                offset_dc,
                picture,
            )
        except Exception as error:
            if index not in self.failed:
                self.failed.add(index)
                logger.error(
                    "The affine on stream %s failed and that stream was left as it was: %s",
                    index,
                    error,
                )
            return None, None


def patch_sampler(sampler, spec: AffineSpec):
    """Wrap a SAMPLER so the affine is injected from inside its own loop.

    Args:
        sampler: A SAMPLER, normally a ``comfy.samplers.KSAMPLER``.
        spec: The affine settings.

    Returns:
        ``(sampler, holder)``. ``holder["mask"]`` receives the last mask applied,
        ``holder["landed"]`` the step and sigma of every application,
        ``holder["gain"]`` what the picture was multiplied by over the whole run,
        ``holder["reach"]`` how much of the latent was picture where it landed,
        ``holder["supported"]`` reports whether the wrap was possible and
        ``holder["reason"]`` says why not. An unwrappable sampler is answered untouched.
    """
    import comfy.samplers

    holder = {
        "mask": None,
        "supported": True,
        "applications": 0,
        "landed": [],
        "gain": 1.0,
        "reach": 1.0,
        "reason": None,
    }

    inner = getattr(sampler, "sampler_function", None)
    if inner is None or not callable(inner):
        holder["supported"] = False
        holder["reason"] = "this sampler has no sampler_function to wrap"
        return sampler, holder

    name = getattr(inner, "__name__", "")
    if name in UNHOOKABLE:
        holder["supported"] = False
        holder["reason"] = UNHOOKABLE[name]
        return sampler, holder
    if spec.is_noop:
        return sampler, holder

    extra_options = dict(getattr(sampler, "extra_options", {}) or {})
    inpaint_options = dict(getattr(sampler, "inpaint_options", {}) or {})

    def affine_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        run = _Run(spec, model, x, sigmas, holder)
        if not run.stream_ids:
            logger.warning(
                "No affine was applied: streams is '%s' but this latent carries %s stream(s), "
                "so nothing was selected. Set streams to 'video' or 'both'.",
                spec.streams,
                len(run.shapes),
            )
        if not run.targets:
            if spec.debug:
                logger.info("No step is scheduled for an affine; the sampler runs untouched.")
            return inner(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **kwargs
            )
        if spec.debug:
            logger.info(
                "Affine over %s steps, landing on %s, latent shapes %s.",
                run.steps,
                sorted(run.targets),
                run.shapes,
            )
        try:
            result = inner(
                run.proxy(model),
                x,
                sigmas,
                extra_args=extra_args,
                callback=run.callback(callback),
                disable=disable,
                **kwargs,
            )
        finally:
            holder["applications"] = run.applications
            holder["landed"] = list(run.landed)
            holder["gain"] = run.gain
            holder["reach"] = (sum(run.reached) / len(run.reached)) if run.reached else 1.0
            run.release()
        missed = sorted(run.targets - run.applied)
        holder["missed"] = missed
        if run.applications == 0:
            logger.warning(
                "No affine was applied: %s step(s) were scheduled but no model call opened "
                "one. Turn on debug for the per-step detail.",
                len(run.targets),
            )
        elif missed:
            logger.warning(
                "The affine skipped step(s) %s: no model call opened them. The other %s "
                "step(s) carried it.",
                missed,
                run.applications,
            )
        return result

    affine_sampler.__name__ = f"affine_{name or 'sampler'}"
    return comfy.samplers.KSAMPLER(affine_sampler, extra_options, inpaint_options), holder
