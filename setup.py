import setuptools

setuptools.setup(
    name = 'flairflow',
    version = "0.0.4",
    author = "Ford Parsons",
    description = "Intercept Flair logging messages, parse them, and log to MLFlow",
    packages = setuptools.find_packages(),
    python_requires = ">=3.7",
    install_requires = ["mlflow"],
)
