#!/bin/sh
basepath=`pwd`

rm -rf ${basepath}/coverage
mkdir -p ${basepath}/coverage


for testdir in `find ./cmake-build-debug/CMakeFiles/Radio_Unit_Tests.dir/ | grep gcda`
do
  testbasedir=$(dirname ${testdir})
  testname=$(basename ${testbasedir})

  echo "Gathering Coverage for ${testname}"
  cd ${testbasedir}


  lcov --capture --directory . --output-file ${testname}.info --quiet --rc lcov_branch_coverage=1

  cd ${basepath}
  if [[ -e all_tests.info ]]
  then
    lcov --add-tracefile all_tests.info --add-tracefile ${testbasedir}/${testname}.info --output-file all_tests.info
  else
    lcov --add-tracefile ${testbasedir}/${testname}.info --output-file all_tests.info
  fi
done

# Only look at my code
lcov --extract all_tests.info '*RadioNode*' -o  filtered.info

cd ${basepath}
genhtml --output-directory ./coverage --demangle-cpp --num-spaces 2 --sort --title "My Program Test Coverage"  --function-coverage --branch-coverage --legend filtered.info
rm -f *.info # remove the info files