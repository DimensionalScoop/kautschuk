# Plugins für ein weniger schlimmes Praktikum

## Atom
In Atom können Plugins in den Einstellungen (*Ctrl+,*) unter *Install* automatisch heruntergeladen und installiert werden. Wenn man ein Plugin ausführen will findet man es meistens in der Palette (*Ctrl+Shift+P*). Hier eine nützliche Auswahl:


atom-beautify

:	Sorgt automatisch für einen schöneren Python-Code indem es z.B. Leerzeichen um Operatoren einfügt. Ist am Anfang etwas gewöhnungsbedürftig, da es den Code umformatiert. Das Plugin orientiert sich an den PEP8 Style Guide. Max empfiehlt folgende Einstellungen im Paket:
	 - Beautify on Save.
	 - Ignore: E24, E502, W293, E303. (damit man noch ein wenig Freiraum bei der Gestaltung des Whitespaces hat, etwa wie viele Zeilen frei bleiben)
	 - Max line lenght: 120 oder 255. (Die voreingestellten 80 Zeichen sind recht kurz und führen zu hässlichen, umgebrochenen Code. 120 ist optimal für gute Lesbarkeit, 255 für Menschen mit langen Zeilen)

autocomplete-python

:	Autovervollständigung mit *Tab* oder *Enter* und Anzeigen von Docstrings/Hilfetexten zu Funktionen und Variablen. Ein Must-Have für Phythonbenutzer.

python-tools
	
:	Hilfreiche Werkzeuge für Pyhton, z.B. Goto Definition.

linter und linter-pylama

:	Hebt Fehler im Pythoncode hervor. Man kann die meisten Syntaxfehler und offensichtliche Fehler (Variable falsch geschrieben, etc.) schon finden, bevor man den Code ausführt. Man muss die Python-Pakete *autopep8* und *pylama* installieren (etwa mit *sudo pip3 install autopep8 pylama*). In den Einstellungen empfiehlt Max unter *Ignore Errors and Warnings* folgende Warnungen einzutragen: *E501,E303,D100*. Dadurch wird man nicht vor selbst gesetzten Whitespace und Dingen, für die man nichts kann, gewarnt.

language-latex

:	Syntax Highlighting für Latex.

latexer

:	Ein paar nützliche Autovervollständigungen (Snippets) für Latex.

script

:	Python- und andere Codedateien direkt in Atom ausführen, kein lästiges Wechseln in das Terminal. Starten mit *Ctrl+Shift+B*. Mit *Ctrl+Q* können Scripte terminiert werden.

minimap

:	Echte Hippster haben sie auf der linken Seite.

activate-power-mode

:	Mit *Ctrl+Alt+O* kann man den Code meißeln statt ihn nur zu schreiben. Erhöht die Motivation nachhaltig.