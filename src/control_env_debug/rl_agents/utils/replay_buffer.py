from collections import deque
import random
import pickle
import gzip
import numpy as np

class ReplayBuffer:
    """
    A general-purpose replay buffer for storing experiences.
    Uses numpy arrays for fast sampling when experience shapes are uniform.
    Falls back to deque-based sampling otherwise.
    """
    def __init__(self, buffer_size):
        """
        Initializes the ReplayBuffer.

        Args:
            buffer_size (int): The maximum number of experiences to store.
        """
        self.buffer_size = int(buffer_size)
        self._pos = 0
        self._full = False
        self._num_fields = None
        # Lazy-init arrays on first add
        self._arrays = None
        # Fallback deque mode for non-uniform data
        self._deque = None

    def _init_arrays(self, experience):
        """Initialize numpy arrays based on the first experience tuple."""
        self._num_fields = len(experience)
        self._arrays = []
        for val in experience:
            val_np = np.asarray(val)
            shape = (self.buffer_size,) + val_np.shape
            arr = np.zeros(shape, dtype=val_np.dtype)
            self._arrays.append(arr)

    def add(self, experience):
        """Adds a single experience to the buffer."""
        if self._deque is not None:
            # Fallback mode
            self._deque.append(experience)
            return
        if self._arrays is None:
            try:
                self._init_arrays(experience)
            except Exception:
                # Fall back to deque
                self._deque = deque(maxlen=self.buffer_size)
                self._deque.append(experience)
                return
        try:
            for j, val in enumerate(experience):
                self._arrays[j][self._pos] = val
        except Exception:
            # Shape mismatch — fall back to deque
            self._deque = deque(maxlen=self.buffer_size)
            # Copy existing data
            n = len(self)
            for idx in range(n):
                self._deque.append(tuple(arr[idx] for arr in self._arrays))
            self._deque.append(experience)
            self._arrays = None
            return
        self._pos += 1
        if self._pos >= self.buffer_size:
            self._full = True
            self._pos = 0

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Returns:
            A list of experience tuples (for backward compat).
        """
        if self._deque is not None:
            return random.sample(self._deque, batch_size)
        n = len(self)
        indices = np.random.randint(0, n, size=batch_size)
        return [self._arrays[j][indices] for j in range(self._num_fields)]

    def sample_arrays(self, batch_size):
        """Sample and return list of numpy arrays directly (no tuple overhead)."""
        if self._deque is not None:
            batch = random.sample(self._deque, batch_size)
            return list(map(lambda x: np.array(x), zip(*batch)))
        n = len(self)
        indices = np.random.randint(0, n, size=batch_size)
        return [self._arrays[j][indices] for j in range(self._num_fields)]

    def __len__(self):
        """Returns the current number of experiences in the buffer."""
        if self._deque is not None:
            return len(self._deque)
        return self.buffer_size if self._full else self._pos

    # New: iteration and export helpers
    def __iter__(self):
        if self._deque is not None:
            return iter(self._deque)
        n = len(self)
        for idx in range(n):
            yield tuple(arr[idx] for arr in self._arrays)

    def as_list(self):
        """Return a shallow list copy of the buffer contents in insertion order."""
        if self._deque is not None:
            return list(self._deque)
        return list(self)

    # Persistence helpers
    def save(self, path: str):
        """Save buffer contents to a gzip-pickled file."""
        data = {
            'maxlen': self.buffer_size,
            'items': self.as_list(),
        }
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        """Load buffer contents from a gzip-pickled file, replacing current contents."""
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        maxlen = int(data.get('maxlen', len(data.get('items', [])) or 0) or 0)
        # Reset to deque mode for loaded data, then re-add
        self.buffer_size = maxlen if maxlen > 0 else len(data.get('items', []))
        self._pos = 0
        self._full = False
        self._arrays = None
        self._deque = None
        self._num_fields = None
        for item in data['items']:
            self.add(item)
