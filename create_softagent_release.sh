RELEASE_DIR=../corl_code_submission/softagent/
CUR_DIR=$(pwd)
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR

rsync -aP --exclude=data/* --exclude=.git --exclude=.idea --exclude=.gitmodule --exclude=softgym  ./ $RELEASE_DIR
cd $RELEASE_DIR

rm -rf DPI-Net PDDM heuristics ResRL dreamer rlkit
rm -rf icml_plot post_ICML_plot corl_plot
rm -rf tests prepare.sh prepare_ec2.sh compile.sh clear.sh pull_s3_result.sh create_softagent_release.sh
rm -rf scripts

rm -rf experiments/dreamer experiments/pddm experiments/model_free experiments/launch_skewfit.py
rm -rf .gitignore

rm -rf chester/config.py chester/config_ec2.py chester/config_private.py