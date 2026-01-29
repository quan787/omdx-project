from setuptools import setup, find_packages

setup(
    name="omdx",
    version="0.1.0",  # 每次发布新版本都要修改这里
    author="quan787",
    author_email="quan787@qq.com",
    description="Open Meteor Data Exchange standard and tools",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/quan787/omdx-project",
    packages=find_packages(),
    license="MPL 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",  # 官方分类器标签
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "opencv-python",
        "astropy",
        "pillow",
        "skyfield",
    ],
    include_package_data=True,
    package_data={
        "omdx": ["data/*.dat", "data/*.bsp", "data/*.json"],
    },
    entry_points={
        "console_scripts": [
            # 格式: 命令名 = 包名.文件名:函数名
            "omdx-player=omdx.player:main",
            "omdx-converter=omdx.converter:main",
        ],
    },
)
