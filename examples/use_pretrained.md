Load the pretrained model.

```python
from deepface import create_deepface, get_weights
model = create_deepface()

weights = get_weights()
model.load_weights(weights)
```
