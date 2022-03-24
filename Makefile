all :
	$(MAKE) -C src
	$(MAKE) -C tests
	$(MAKE) -C ocr/lib
	$(MAKE) -C ocr

clean :
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
	$(MAKE) -C ocr/lib clean
	$(MAKE) -C ocr clean

clean-all :
	$(MAKE) -C src clean-all
	$(MAKE) -C tests clean-all
	$(MAKE) -C ocr/lib clean-all
	$(MAKE) -C ocr clean-all

version :
	$(MAKE) -C src version

source-dist :
	$(MAKE) -C ocr network1.net network2.net
	$(MAKE) -C src src-dist
	$(MAKE) -C tests src-dist
	$(MAKE) -C ocr/lib src-dist
	$(MAKE) -C ocr src-dist
	(VERSION=`cat VERSION` ; \
	tar --create --directory .. --exclude CVS --exclude .cvsignore \
	    --exclude ocrdata --exclude website --exclude private \
	    -f /tmp/nerveNet-$${VERSION}.tar.gz \
	    --gzip --verbose nerveNet-$${VERSION} ; \
	mv /tmp/nerveNet-$${VERSION}.tar.gz .)



