# flairflow
Intercept Flair logging messages, parse them, and log to MLflow

## Background

[Flair](https://github.com/flairNLP/flair/) is an amazing NLP library.
[MLflow](https://mlflow.org/) is an incredibly useful tool to keep track
of machine learning parameters, metrics, and artifacts. However, Flair doesn't currently support MLflow, nor does it support a callback
mechanism to get updates about training metrics.

The `flairflow` library is an attempt to work around these limitations
and enable MLflow logging in Flair.

It accomplishes this by adding a log handler to the logger used by Flair, intercepting and parsing these log messages, and forwarding them to MLflow.
It's by no means elegant, but it works.

## Example Usage
```python
import flair, mlflow
from flairflow import FlairLogMLFLow

# set up Flair's trainer object...

# flairflow will log to mlflow for all flair messages
# in the context of this `with` statement
with mlflow.start_run(), FlairLogMLFLow():
    result = trainer.train(base_path=path,
                           learning_rate=0.1,
                           mini_batch_size=32,
                           max_epochs=150)
```
