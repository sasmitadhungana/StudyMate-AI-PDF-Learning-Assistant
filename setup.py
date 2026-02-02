from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="studymate-ai",
    version="1.0.0",
    author="StudyMate AI Team",
    author_email="your-email@example.com",
    description="AI-powered PDF Learning Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sasmitadhungana/StudyMate-AI-PDF-Learning-Assistant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "PyPDF2>=3.0.1",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "studymate-ai=src.frontend.app:main",
        ],
    },
)
