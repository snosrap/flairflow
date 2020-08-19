import logging, io, re, mlflow

class FlairLogInterceptor(io.StringIO):
    """
    Gets the logger instance named 'flair', adds itself as a stream handler, and listens
    to Flair's logging by overriding the `write(s)` method of `io.StringIO`
    This base class contains empty methods: on_epoch, on_dev, on_f1_micro, on_f1_macro, on_label
    Usage:
    from flairflow import FlairLogInterceptor
    with mlflow.start_run(), FlairLogInterceptor():
        result = trainer.train(...) # do flair processing here
    """
    re_epoch = re.compile(r"EPOCH (\d+) done: loss ([\d\.]+) - lr ([\d\.]+)")
    re_dev_loss = re.compile(r"DEV : loss ([\d\.]+) - score ([\d\.]+)")
    re_f1_micro = re.compile(r"- F1-score \(micro\) ([\d\.]+)")
    re_f1_macro = re.compile(r"- F1-score \(macro\) ([\d\.]+)")
    re_label = re.compile(r"(\w+)\s*tp: (\d+) - fp: (\d+) - fn: (\d+) - precision: ([\d\.]+) - recall: ([\d\.]+) - f1-score: ([\d\.]+)")

    def __enter__(self):
        self.log = logging.getLogger("flair")
        self.handler = logging.StreamHandler(self)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.handler.setLevel(logging.INFO)
        self.log.addHandler(self.handler)

    def __exit__(self, type, value, traceback):
        self.log.removeHandler(self.handler)

    def on_epoch(self, epoch_num, train_loss, train_lr): pass
    def on_dev(self, dev_loss, dev_score): pass
    def on_f1_micro(self, f1_micro): pass
    def on_f1_macro(self, f1_macro): pass
    def on_label(self, label, tp, fp, fn, precision, recall, f1): pass

    def write(self, s):
        for match in self.re_epoch.findall(s):
            epoch_num, train_loss, train_lr = match
            self.on_epoch(int(epoch_num), float(train_loss), float(train_lr))
        for match in self.re_dev_loss.findall(s):
            dev_loss, dev_score = [float(x) for x in match]
            self.on_dev(dev_loss, dev_score)
        for match in self.re_f1_micro.findall(s):
            f1_micro = float(match)
            self.on_f1_micro(f1_micro)
        for match in self.re_f1_macro.findall(s):
            f1_macro = float(match)
            self.on_f1_macro(f1_macro)
        for match in self.re_label.findall(s):
            label, tp, fp, fn, precision, recall, f1 = match
            self.on_label(label, int(tp), int(fp), int(fn), float(precision), float(recall), float(f1))
        return super().write(s)

class FlairLogMLFLow(FlairLogInterceptor):
    """
    Subclass of FlairLogInterceptor that actually sends the parsed metrics to MLFlow
    Usage:
    from flairflow import FlairLogMLFLow
    with mlflow.start_run(), FlairLogMLFLow():
        result = trainer.train(...) # do flair processing here
    """
    def on_epoch(self, epoch_num, train_loss, train_lr):
        mlflow.log_metric('epoch_num', epoch_num)
        mlflow.log_metric('train_loss', train_loss)
        mlflow.log_metric('train_lr', train_lr)

    def on_dev(self, dev_loss, dev_score):
        mlflow.log_metric('dev_loss', dev_loss)
        mlflow.log_metric('dev_score', dev_score)

    def on_f1_micro(self, f1_micro):
        mlflow.log_metric('f1_micro', f1_micro)

    def on_f1_macro(self, f1_macro):
        mlflow.log_metric('f1_macro', f1_macro)

    def on_label(self, label, tp, fp, fn, precision, recall, f1):
        mlflow.log_metric(f'{label}_tp', tp)
        mlflow.log_metric(f'{label}_fp', fp)
        mlflow.log_metric(f'{label}_fn', fn)
        mlflow.log_metric(f'{label}_precision', precision)
        mlflow.log_metric(f'{label}_recall', recall)
        mlflow.log_metric(f'{label}_f1', f1)
