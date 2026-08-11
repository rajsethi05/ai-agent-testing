def log_metrics(**kwargs) -> str:
    metrics = ""
    for metric, value in kwargs.items():
        if type(value) is float:
            value = round(value, 5)
        metrics = metrics + f' {metric}={value},'
    return metrics
