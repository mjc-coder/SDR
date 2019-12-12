



# Define where QMAKE lives
QMAKE=/Users/michealcowan/Qt/5.13.1/clang_64/bin/qmake

all: clean runRadioNode dox
	@echo "Build Complete"

clean:
	@rm -rf dst
	@echo "Cleaning ..."


mkdst:
	mkdir -p dst
	mkdir -p dst/RadioNode
	mkdir -p dst/doc/RadioNode
	mkdir -p dst/doc/ThirdParty_FirFilter
	mkdir -p dst/doc/ThirdParty_hackrf
	mkdir -p dst/doc/ThirdParty_RtlSdr
	mkdir -p dst/doc/warnings


buildRadioNode: mkdst
	cmake -B./dst/RadioNode -H./RadioNode
	make -C ./dst/RadioNode
	@echo "Radio Node Build Complete ..."


runRadioNode: buildRadioNode
	@echo "Running RadioNode ..."
	./dst/RadioNode/Radio_Unit_Tests
	./RadioNode/generate_coverage.sh

dox: mkdst
	doxygen doc/RadioNode.dox
	make -C dst/doc/RadioNode/latex
	./doc/generate_appendix.sh dst/doc/RadioNode/latex doc/wsu_ece_thesis/tex/Appendix_RadioNode.tex
	doxygen doc/ThirdParty_RTLSDR.dox
	make -C dst/doc/ThirdParty_RtlSdr/latex
	./doc/generate_appendix.sh dst/doc/ThirdParty_RtlSdr/latex doc/wsu_ece_thesis/tex/Appendix_ThirdParty_RtlSdr.tex
	doxygen doc/ThirdParty_hackrf.dox
	make -C dst/doc/ThirdParty_hackrf/latex
	./doc/generate_appendix.sh dst/doc/ThirdParty_hackrf/latex doc/wsu_ece_thesis/tex/Appendix_ThirdParty_hackrf.tex
	doxygen doc/ThirdParty_FIR_filter-class.dox
	make -C dst/doc/ThirdParty_FirFilter/latex
	./doc/generate_appendix.sh dst/doc/ThirdParty_FirFilter/latex doc/wsu_ece_thesis/tex/Appendix_ThirdParty_FirFilter.tex
	@echo "Doxygen builds complete ..."
	@echo "Generating Software Design Documentation ..."
	make -C doc/wsu_ece_thesis
	@echo "========================================================================================="
	@echo "Doxygen Warnings"
	@wc -l dst/doc/warnings/*warnings.txt
	@echo "Documentation warnings should be resolved to generate complete Software Documentation"
	@echo "========================================================================================="


thesis:
	@echo "Generating Software Design Documentation ..."
	make -C doc/wsu_ece_thesis



