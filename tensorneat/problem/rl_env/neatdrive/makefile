build:
	easycython cythonfib.py
	easycython *.pyx
	

clean:
	@echo "Passando a vassoura"
	del /s /q __pycache__
	del /q *.so
	del /q f*.c
	rmdir /s /q build
	del /q *.html
	del /q prof*
