# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


from .wan_ti2v_5B import ti2v_5B


WAN_CONFIGS = {
    'ti2v-5B': ti2v_5B,
}

SIZE_CONFIGS = {
    '2560*1440': (2560, 1440),
    '3840*2144': (3840, 2144),
    '448*256': (448, 256),
}

MAX_AREA_CONFIGS = {
    '2560*1440': 2560 * 1440,
    '3840*2144': 3840 * 2144,
    '448*256': 448 * 256,
}

SUPPORTED_SIZES = {
    'ti2v-5B': (
        '448*256',
        '2560*1440',
        '3840*2144',
    ),
}
