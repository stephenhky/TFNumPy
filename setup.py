
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


def package_description():
    text = readme()
    startpos = text.find('# Numerical Python using TensorFlow')
    return text[startpos:]


setup(name='tfnumpy',
      version="0.1.0",
      description="Collection of Simple Numerical Routines using TensorFlow",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Mathematics",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
      ],
      keywords="numerics computation tensorflow",
      url="https://github.com/stephenhky/TFNumPy",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['tfnumpy',
                'tfnumpy.embed',
                'tfnumpy.regression',
                'tfnumpy.utils',
                'tfnumpy.tensor',
                ],
      install_requires=[
          'numpy', 'tensorflow>=2.0.0',
      ],
      tests_require=[
          'unittest2', 'scikit-learn',
      ],
      scripts=['bin/tfsammon'],
      include_package_data=True,
      test_suite="test",
      zip_safe=False)

