import os, logging

logging.basicConfig(level="WARNING")

import numpy as np
import fairseq
import pydub

import pyrvc.module as module
from .wave import Wave


class Model():
    _hubert_model = None

    def __init__(self, model_file: str) -> None:
        self._model = module.torch.load(model_file, map_location="cpu")
        keys = [
            "spec_channels", "segment_size", "inter_channels", "hidden_channels",
            "filter_channels", "n_heads", "n_layers", "kernel_size", "p_dropout",
            "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
            "upsample_rates", "upsample_initial_channel",
            "upsample_kernel_sizes", "spk_embed_dim", "gin_channels", "sr"]
        for i, key in enumerate(keys): self._model["params"][key] = self._model["config"][i]
        self._model["params"]["spk_embed_dim"] = self._model["weight"]["emb_g.weight"].shape[0]
        self.sr = self._model["params"]["sr"]
        if_f0 = self._model.get("f0", 1) == 1
        if if_f0: self._net_g = module.SynthesizerTrnMs256NSFSid(**self._model["params"], is_half=module.is_half)
        else: self._net_g = module.SynthesizerTrnMs256NSFSidNono(**self._model["params"])
        del self._net_g.enc_q
        self._net_g.load_state_dict(self._model["weight"], strict=False)
        self._net_g.eval().to(module.device)
        self._net_g = self._net_g.half() if module.is_half else self._net_g.float()
        self._n_spk = self._model["params"]["spk_embed_dim"]
        self._converter = module.Converter(self.sr, if_f0)
        if self._hubert_model is None:
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([os.path.join(os.path.dirname(__file__), "hubert_base.pt")], suffix="")
            self._hubert_model = models[0].to(module.device).half() if module.is_half else models[0].to(module.device).float()
            self._hubert_model.eval()

    def convert(self, wave: Wave, *, raise_pitch: int=0, f0_method: str="pm"):
        """Converts wave data by the loaded model.
        ``wave: np.ndarray`` - wave data of float (-1.0 ~ 1.0)
        ``sr: int`` - sampling rate of the wave data. 16000 is preferred.
        ``raise_pitch: int=0`` - change the pitch of output by the value.
        ``f0_method: str="pm"`` - ``"pm"`` or ``"harvest"``
        ``return: np.ndarray(int16)``
        """
        wave = wave.change_sr_(16000)
        times = [0, 0, 0]
        a = self._converter(self._hubert_model, self._net_g, 0, wave, times, raise_pitch, f0_method)
        return Wave.from_numpy(a, self.sr)
