from setuptools import setup, find_packages
import os

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='glm_utils',
      version='0.1',
      description='glm_utils',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/janclemenslab/glm_utils',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['scikit-learn', 'numpy'],
      include_package_data=True,
      zip_safe=False
      )
