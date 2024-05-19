echo '> freezing requirements'
echo '>> lines before'
wc -l requirements.txt
pip freeze | grep -v "^\-e" >requirements.txt
echo '>> lines after'
wc -l requirements.txt
echo ''
echo '> installing -e .'
pip install -e . >/dev/null
