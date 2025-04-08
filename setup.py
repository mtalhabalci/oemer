import sys
import setuptools


with open("README.md") as red:
    ldest = red.read()


if sys.platform == "darwin":
    onnx_package = "onnxruntime"
else:
    onnx_package = "onnxruntime-gpu"


setuptools.setup(
    name="oemer",
    version="0.1.8",
    author="BreezeWhite",
    author_email="miyasihta2010@tuta.io",
    description="End-to-end Optical Music Recognition (OMR) system.",
    long_description=ldest,
    long_description_content_type="text/markdown",
    license="License :: OSI Approved :: MIT License",
    license_files=("LICENSE",),
    url="https://github.com/BreezeWhite/oemer",
    packages=setuptools.find_packages(),
    package_data={
        "": [
            "sklearn_models/*.model",
            "checkpoints/unet_big/metadata.pkl",
            "checkpoints/unet_big/arch.json",
            "checkpoints/seg_net/metadata.pkl",
            "checkpoints/seg_net/arch.json",
        ]
    },
    install_requires=[
        onnx_package,
        "opencv-python-headless>=4.5.3.56",
        "matplotlib",
        "pillow",
        "scipy",
        "scikit-learn>=1.2",
        "types-Pillow",
        "types-tensorflow",
        "typing-extensions",
    ],
    extras_require={
        "tf": ["tensorflow", "tf2onnx"],
    },
    entry_points={"console_scripts": ["oemer = oemer.ete:main"]},
    keywords=[
        "OMR",
        "optical-music-recognition",
        "AI",
        "machine-learning",
        "image-processing",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
    ],
)
