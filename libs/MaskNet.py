import os
import sys
import apt
from importlib import import_module
from importlib.util import find_spec

requirements = {
    "pip": {
        "GitPython": {
            "version": None
        },
        "numpy": {
            "version": None
        }
    },
    "apt": {
        "fbi": {
            "version": "2.10-3"
        }
    },
    "check_only": {
        "OpenCV": {
            "python_module":
            True,
            "py_module_name":
            "cv2",
            "version":
            "4.5.0",
            "error": [
                "OpenCV version 4.5.0 is not installed.",
                "Compile from: https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html"
            ]
        },
        "TensorFlow": {
            "python_module":
            True,
            "py_module_name":
            "tensorflow",
            "version":
            "2.4.0",
            "error": [
                "TensorFlow version 2.4.0 is not installed.",
                "Compile from: https://qengineering.eu/install-tensorflow-2.4.0-on-raspberry-64-os.html"
            ]
        }
    }
}


def check_requirements():
    python_version = f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}'
    if sys.version_info[0] != 3 or (sys.version_info[0] == 3
                                    and sys.version_info[1] < 6):
        raise Exception(
            f"Python version is {python_version}, but MaskNet requires python version to be 3.6.0 or higher."
        )
    print("[Step 1] Required Manually Compiled Packages:")
    for check_only_package in requirements["check_only"].items():
        if check_only_package[1]["python_module"]:
            print(f"Checking if '{check_only_package[0]}' is installed...")
            python_module_name = check_only_package[1]["py_module_name"]
            found = find_spec(python_module_name)
            if found is not None:
                module = import_module(python_module_name)
                installed_version = module.__version__
                if installed_version != check_only_package[1]["version"]:
                    raise Exception(
                        f"Module '{check_only_package[0]}' version {installed_version} is installed, but MaskNet requires version {check_only_package[1]['version']}.\nPlease run `pip uninstall {check_only_package[0]}={check_only_package[1]['version']}` and {check_only_package[1]['error'][1].lower()}"
                    )
                del module
            else:
                raise Exception("\n".join(
                    map(str, check_only_package[1]["error"])))

    print("\n[Step 2] Required Apt Packages:")
    for apt_package in requirements["apt"].items():
        print(
            f"Checking if '{apt_package[0]}' (version {apt_package[1]['version']}) is installed..."
        )
        cache = apt.Cache()
        if cache[apt_package[0]].is_installed:
            print(f"'{apt_package[0]}' is installed!")
        else:
            answer = "None"
            while answer not in ["y", "n"]:
                answer = input(
                    f"'{apt_package[0]}' is not installed. Do you want to install it now [Y/n]? "
                ).lower()
                if len(answer) == 0:
                    break
            if answer not in ["None", "n"]:
                try:
                    import subprocess
                    subprocess.check_call([
                        'sudo', 'apt', 'install', '-y', '-qq', apt_package[0]
                    ])
                    print(f"'{apt_package[0]}' is installed!")
                except:
                    raise Exception(
                        f"'{apt_package[0]}' failed to install.\nTry to install using `sudo apt install -y {apt_package[0]}`."
                    )

    print("\n[Step 3] Required Pip Packages:")
    for pip_package in requirements["pip"].items():
        print(
            f"Checking if '{pip_package[0]}' (version {pip_package[1]['version']}) is installed..."
        )
        if pip_package[0] in sys.modules:
            print(f"'{pip_package[0]}' is installed!")
        else:
            answer = "None"
            if answer not in ["y", "n"]:
                answer = input(
                    f"'{pip_package[0]}' is not installed. Do you want to install it now [Y/n]? "
                ).lower()
                if len(answer) == 0:
                    break
            if answer not in ["None", "n"]:
                try:
                    import subprocess
                    subprocess.check_call(
                        ['pip', 'install', '-q', pip_package[0]])
                    print(f"'{pip_package[0]}' is installed!")
                except:
                    raise Exception(
                        f"'{pip_package[0]}' failed to install.\nTry to install using `pip install {pip_package[0]}`."
                    )


check_requirements()
