import array
import threading

import numpy as np
import pydub
from pydub.playback import play

DTYPE2TYPECODE = {
    np.int8: "b",
    np.int16: "h",
    np.int32: "l",
    np.int64: "q",
    np.float32: "f",
    np.float64: "d",
}

class Wave():
    def __init__(self, data, sr, channels, *args):
        """Initialize from data and sampling rate.
        ``data: 1D array of int16``
        ``sr: int`` - sampling rate
        ``channels: int``
        """
        if data is None: self.__as = pydub.AudioSegment.from_file(args[0])
        else: self.__as = pydub.AudioSegment(data, sample_width=2, frame_rate=sr, channels=channels)
        self.__samples = self.__as.frame_count()

    @classmethod
    def from_file(cls, file):
        return Wave(None, None, None, file)

    @classmethod
    def from_numpy(cls, ndarray: np.ndarray, sr: int):
        assert ndarray.dtype == np.int16
        if len(ndarray.shape) == 1: return Wave(ndarray.tobytes(), sr, 1)
        else: return Wave(ndarray.tobytes(), sr, ndarray.shape[-1])

    @property
    def sr(self): return self.__as.frame_rate

    @property
    def samples(self): return self.__samples

    @property
    def channels(self): return self.__as.channels

    @property
    def audio_segment(self): return self.__as

    def asnumpy(self, dtype=np.int16):
        return np.array(self.__as.get_array_of_samples(), dtype).reshape(-1, self.channels)

    def save_as(self, file: str, format: str=None):
        """Saves this wave as the specified file format.
        ``file: str`` - file path
        ``format: str=None`` - from ``file`` if is None.
        """
        if format is None: format = file.split(".")[-1]
        self.__as.export(file, format)

    def play(self, background: bool=False):
        """Plays this wave by pyaudio.
        ``background: bool=False``
        """
        if background: threading.Thread(target=lambda: play(self.__as)).start()
        else: play(self.__as)

    def monaural(self):
        if self.channels <= 1: return self
        return Wave.from_numpy((self.asnumpy(np.int32).sum(-1)/2).astype(np.int16), self.sr)

    def change_sr_(self, sr: int) -> np.ndarray:
        """Changes sampling rate and returns ndarray.
        ``sr: int`` - sampling rate
        ``return: ndarray(float64)`` - monaural
        """
        a = np.average(self.asnumpy(np.float64), -1)
        n = int(self.samples * sr / self.sr)
        x = np.arange(n, dtype=np.float64) * (self.sr / sr)
        return np.interp(x, np.arange(self.samples), a)

    # def change_sr(self, sr: int) -> np.ndarray:
    #     """
    #     ``return: np.ndarray(np.float64)``
    #     """
    #     if self.interpolate is None:

    #         y = self.asnumpy(np.float_)
    #         # self.interpolators = [interpolate.interp1d(x, y[:,i]) for i in range(self.channels)]
    #     if sr == self.sr: return self
    #     n = int(self.samples * sr / self.sr)
    #     x = np.arange(n, dtype=np.float_) * (self.sr / sr)
    #     y = np.zeros((n, self.channels), np.float_)
    #     for i, interpolator in enumerate(self.interpolators): np.copyto(y[:,i], np.interp(x, x = np.arange(self.samples), ))
    #     return Wave.from_numpy(y.astype(np.int16), sr)
