#!/usr/bin/env python3

import deepface
model = deepface.create_deepface()

weights = deepface.get_weights()
model.load_weights(weights)
