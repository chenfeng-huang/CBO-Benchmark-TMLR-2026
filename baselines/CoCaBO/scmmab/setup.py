from distutils.core import setup

setup(
    name='npsem',
    packages=['npsem', 'npsem.experiments'],
    version="0.1.0",
    author='***',
    author_email='*@gmail.com', requires=['numpy', 'scipy', 'joblib', 'matplotlib', 'seaborn', 'networkx']
)
