CONFIDR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $CONFDIR/test-conda-pkgs.sh
for sc in $COSMOINSTALL ; do
    if [[ ! -z $sc ]] ; then
        source $CONFDIR/$sc
    fi
done
