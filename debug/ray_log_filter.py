import os
import re
import sys

# apt-get install rlwrap
# rlwrap python pipeline.py


class RayPrefixFilter:
    """Filter (ActorName pid=XXXX) prefixes from Ray logs."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        # Pattern to match (RayWorkerWrapper pid=123) or (Actor pid=123, ip=...)
        self.prefix_pattern = re.compile(r'\([^)]*(?:pid=\d+|ip=[^)]+)[^)]*\)\s*')

    def write(self, data):
        self.buffer += data
        lines = self.buffer.split('\n')
        self.buffer = lines[-1]  # Keep incomplete line in buffer

        for line in lines[:-1]:
            cleaned = self.prefix_pattern.sub('', line)
            self.stream.write(cleaned + '\n')

    def flush(self):
        if self.buffer:
            cleaned = self.prefix_pattern.sub('', self.buffer)
            self.stream.write(cleaned)
            self.buffer = ''
        self.stream.flush()

    # Delegate other attributes to original stream
    def __getattr__(self, name):
        return getattr(self.stream, name)


# Apply the filter BEFORE importing ray
sys.stdout = RayPrefixFilter(sys.stdout)
sys.stderr = RayPrefixFilter(sys.stderr)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['LMDEPLOY_SKIP_WARMUP'] = '1'
    os.environ['RAY_DEDUP_LOGS'] = '0'  # Disable "repeated X times" messages
