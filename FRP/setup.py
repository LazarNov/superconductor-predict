from setuptools import setup, find_packages

setup(
    name='ASCARF',
    version='1.0',
    description='Atoms and structures described distribution density',
    author='L Novakovic',
    author_email='lazarnovakovic3@gmail.com',
    url='https://github.com/LazarNov/ASCARF',
    packages=find_packages(),
    install_requires=[
    'pandas==1.3.5',
        'scipy==1.7.3','tensorflow==2.11.0',
    
    'sklearn',
    
        'requests',
        'numpy==1.21.0',
        'pymatgen',
        'chemparse>=0.1.2',
         'keras-tuner==1.1.3',
        'mendeleev==0.12.1',
         'aflow==0.0.11', 'keras-complex', 'cvnn', 'tqdm', 'rpy2==3.4.2'

    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
       
    ],
)
