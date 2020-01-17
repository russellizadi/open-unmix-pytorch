#!/usr/bin/env bash

NAME_REPO="open-unmix" # venv name
NAME_CUDA="cuda-10.0" # cuda version
FILE_PIP="requirements.txt" # requirements

PATH_VENV="${HOME}/.venv" # venv installation
PATH_CUDA="/usr/local"  # cuda installations

# forget other commands
set -e

# ensure that the script has been sourced rather than just executed
if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
    echo "Run '$ source setup.sh'"
    exit 1
fi

# ensure that python3 is installed
if ! [ -x "$(command -v python3)" ]; then
  echo 'Error: python3 is not installed.' >&2
  set +e
  return
fi

# list available cuda versions 
if [[ -z ${NAME_CUDA} ]]; then
    echo "The following CUDA installations have been found in '${PATH_CUDA}'):"
    ls -l "${PATH_CUDA}" | egrep -o "cuda-[0-9]+\\.[0-9]+$" | while read -r line; do
        echo "* ${line}"
    done
    set +e
    return
# otherwise, check whether there is an installation
elif [[ ! -d "${PATH_CUDA}/${NAME_CUDA}" ]]; then
    echo "Warning: No installation of CUDA ${NAME_CUDA} has been found!"
    set +e
fi

# filter out non CUDA paths
path_cuda="${PATH_CUDA}/${NAME_CUDA}"
lst_paths=(${PATH//:/ })
path="${path_cuda}/bin"

for p in "${lst_paths[@]}"; do
    if [[ ! ${p} =~ ^${PATH_CUDA}/cuda ]]; then
        path="${path}:${p}"
    fi
done

# filter out non CUDA ld paths 
lst_paths_ld=(${LD_LIBRARY_PATH//:/ })
path_ld="${path_cuda}/lib64:${path_cuda}/extras/CUPTI/lib64"
for p in "${lst_paths_ld[@]}"; do
    if [[ ! ${p} =~ ^${PATH_CUDA}/cuda ]]; then
        path_ld="${path_ld}:${p}"
    fi
done

# update environment variables
export CUDA_HOME="${path_cuda}"
export CUDA_ROOT="${path_cuda}"
export LD_LIBRARY_PATH="${path_ld}"
export PATH="${path}"

# check pkgs
LST_PKG="python3-pip python3-venv"
for pkg in $LST_PKG; do
	if [ $(dpkg-query -W -f='${Status}' $pkg 2>/dev/null | grep -c "ok installed") -eq 0 ]; then
		echo 'Eroor:' $pkg 'is not installed, run apt-get install' $pkg >&2
  		set +e
  		return
	fi
done

# alias for ease
alias python=python3
alias pip=pip3

if test ! -d $PATH_VENV/$NAME_REPO;
then python -m venv $PATH_VENV/$NAME_REPO;
fi

echo 'export OLD_PYTHONPATH="$PYTHONPATH"' >> \
"${PATH_VENV}/${NAME_REPO}/bin/activate"

echo 'export PYTHONPATH="$PWD"' >> "${PATH_VENV}/${NAME_REPO}/bin/activate"

echo 'export PYTHONPATH="$OLD_PYTHONPATH"' >> \
"${PATH_VENV}/${NAME_REPO}/bin/postactivate"

source $PATH_VENV/$NAME_REPO/bin/activate

if test -f $FILE_PIP; then
	pip -q install -r $FILE_PIP
fi

set +e
return
