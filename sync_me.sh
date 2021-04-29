rsync -av \
    --delete-before \
    --exclude "persistent_storage/text2topic/app/data/processed" \
    $PWD/ \
    /media/jamie/HDD/adv_project/
