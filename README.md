# Kautschuk
A `Latex`-Scrip collection for physic's laboratory class and all other concerns.

# Folder structure and prerequisites
## AP_SS16
For every different experiment we have a single folder with the name of the experiment's number, eg. "402". These experiments can be found in ordered folders, eg. AP_SS16 ("Anfängerpraktikum"). Each folder contains another folder named "AP_SS16/402/final". Here we've put in our final version of the concerning protocol. Other contents come from the folder "latex-template". The files in there are your template for every protocol. You find all the custom python scripts and the latex files in here.
Also you will see a Makefile which you should use in order to build your protocol. We use Python 3.4, Lualatex and Biber, so make sure you have installed these programs. Here is a nice link to do so:
http://toolbox.pep-dortmund.org/install.html

## Die halbe Trommel zum Glück
Physicians will be glad to see this :)

## Editor Guide
Find out how to optimize your atom text editor.

## kappa
Don't ask, just have in mind that this folder exists if you are annoyed of your greek kappa coming with latex.

## latex-template
### Makefile
What does the makefile do? First it will create another folder named "build". PythonSkript.py will be evaluated now. This file is the heart of the calculations. It will create files which will be included in the .tex project. Afterwards lualatex and biber will be run. Options for lualatex will be

```{r, engine='bash', count_lines}
latexmk \
  --lualatex \
  --output-directory=build \
  --interaction=nonstopmode \
  --halt-on-error \
  --synctex=1 \
```

We have taken care that our makefiles will work if you have the needed programs installed. For example, in order to build experiment 402 navigate to this folder in your console and type "make".

### PythonSkript.py
As mentioned, this file is the heart of the concerning protocol's calculations. In the head of this file there are severel imports, make sure you have installed the needed packages. For example you can use
```{r, engine='bash', count_lines}
conda install scipy
```
if you have Anaconda installed and want to install scipy (what you really should do!). Following to the header is our specific code for the protocol. We tried to structure it a little bit and wrote some comments. It could be worthy to take a look at "./StyleGuide.pdf".
Many code snippets are used over and over again. We have made a list of those snippets which can be found at the bottom of PythonSkript.py. These examples are pretty helpful so consider to take a look at them! Also there are examples of how to use our custom scripts. Biggest advantage is the usage of make_full_table in order to avoid annoying table generation in latex. Note that for plotting we use Latex code in order to make everything nice. This will be done using "latex-template/meine-matplotlibrc" which can of course also be modified.

### Protokoll.tex
Your Latex project's main file is "latex-template/Protokoll.tex". Here all the other files will be included starting with header.tex. Of course you can make changes to "latex-template/header.tex" in order to customize your output. The other includes are the content files found in "latex-template/content". Write your literary masterpieces here.

## spell checker
Informations on how to set up a spell checker.

## Style Guide
Within this folder you find our style guide for writing python code. We have found that a common style can be pretty helpful in order to have comparable code.

## tex-table-generator
Here we develop functions aiming towards automatic generation of latex tables. Also you will find the documentation for these functions (especially table.py) in the subfolder "Doku table_py".
