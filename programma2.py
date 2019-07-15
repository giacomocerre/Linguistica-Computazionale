# -*- coding: utf-8 -*-
import sys,codecs,nltk,math
from nltk import bigrams
from nltk import trigrams
from math import log

def CorpusTokensPOS (frasi):
	lista_tokens = []
	lista_POS_totali = []
	lista_POS_finale = []
	i = 0
	# ciclo per la tokenizzazzione
	for frase in frasi:
		tokens = nltk.word_tokenize(frase.encode('utf8')) # tokenizzo la frase codificta in utf-8
		lista_tokens = lista_tokens + tokens # creo la lista completa di tutti i token
	corpus = len(lista_tokens)
	POS_token_tag = nltk.pos_tag(lista_tokens) # esegue l'assegnamento del POS al token
	return corpus, lista_tokens, POS_token_tag

def analisiFrequenze(tokens, tokens_POS, POS):
	# TOKEN
	lista_senza_punteggiatura = estraiSenzaPunteggiatura(tokens) # richiamo della funzione per invertire l'ordine della lista e togliere le frequenze della punteggiatura
	token_20 = ordina(conta(lista_senza_punteggiatura))
	# AGGETTIVI E VERBI
	lista_Aggettivi = estraiTipiGrammaticali(tokens_POS, ["JJ", "JJR", "JJS"]) # estraggo gli aggettivi
	lista_Verbi = estraiTipiGrammaticali(tokens_POS, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) # estraggo i verbi
	aggettivi_20 = ordina(conta(lista_Aggettivi))
	verbi_20 = ordina(conta(lista_Verbi))
	# POS 
	POS_10 = ordina(conta(POS))
	# TRIGRAMMI DI POS
	lista_trigrammi = trigrams(POS)
	trigrammi_10 = analisiN_grammi(lista_trigrammi)
	return token_20[:20], aggettivi_20[:20], verbi_20[:20], POS_10[:10], trigrammi_10

def analisiN_grammi (lista):
	lista_tot_frq_POS = [] # lista totale bigrammi <frequenza, POS_tag>
	distribuzione_di_frequenza = nltk.FreqDist(lista) # distribuzione di frequenza degli elementi all'interno della lista
	# ciclo per creare una lista contenente le varie associazioni <frequenza, POS_tag>
	for tag in distribuzione_di_frequenza:
		lista_tot_frq_POS.append([distribuzione_di_frequenza[tag], tag])
	lista_tot_frq_POS.sort(reverse = True) # ordino e inverto la lista
	# return primi 10 elementi della lista
	return lista_tot_frq_POS[:10]

# Funzione che calcola la probabilità condizionata e congiunta di un bigramma di PoS
def probabilita (lista_POS, lista_bigrammi):
	lista_bigramma_PCONG = []
	lista_bigramma_PCOND = []
	N = len(lista_POS)
	distribuzione_di_frequenza = nltk.FreqDist(lista_bigrammi) # distribuzione di frequenza degli elementi all'interno della lista
	# cilco per calcolo delle probabilità congiunte e condizionate sui bigrammi di PoS
	for bigramma in distribuzione_di_frequenza:
		# Frequenza osservata F(a,b)
	 	frequenza_osservata = distribuzione_di_frequenza[bigramma]
	 	# Frequenza attesa F(a) * F(b)/ N
	 	frequenza_a = lista_POS.count(bigramma[0])
		frequenza_b = lista_POS.count(bigramma[1])
		frequenza_attesa = frequenzaAttesa(frequenza_a, frequenza_b, N)
		# Calcolo Probabilità congiunta
	 	P_CONG = (((frequenza_osservata*1.0)/(N*1.0))*100)
	 	# Calcolo Probabilità condizionata
		P_COND = (frequenza_attesa/N*1.0)*100
	 	lista_bigramma_PCONG.append([P_CONG, bigramma])
	 	lista_bigramma_PCOND.append([P_COND, bigramma])
	lista_bigramma_PCONG = ordina(lista_bigramma_PCONG) # ordino le liste in ordine decrescente in base alla loro probabilità
	lista_bigramma_PCOND = ordina(lista_bigramma_PCOND)
	# return primi 10 elementi
	return lista_bigramma_PCONG[:10], lista_bigramma_PCOND[:10]

# Funziuone che cacola la frequenza attesa di un bigramma
def frequenzaAttesa (a, b, N):
	# F(a) * F(b)/ N
	tot = ((a*1.0)*(b*1.0))/(N*1.0)
	return tot

# Funzione che estrae i bigrammi e successivamente crea una sola lista composta dai 10 sostantivi piu frequenti del file1 e del file2
def listaAggSost(tokens_POS1, tokens_POS2):
	lista_SOST1 = estraiSOST(tokens_POS1)
	lista_SOST2 = estraiSOST(tokens_POS2)
	lista_unica = lista_SOST1  # creo la lista unica 
	lista_uguali = [] # lista di possibili SOST uguali nei due corpus
	# ciclo che confronta gli elementi della seconda lista con la pirma lista e unisce solo i valori diversi.
	for elemento in lista_SOST2:
		if elemento not in lista_unica:
			lista_unica.append(elemento)
		else:
			lista_uguali.append(elemento)
	return lista_unica, lista_uguali

def estraiSOST(token_POS):
	lista_SOST = []
	lista_ordinata = []
	for elemento in token_POS:
		if elemento[1] in ["NN", "NNS"]: # estraggo solo i bigrammi composti da la coppia <AGGETTIVO,SOSTANTIVO>
			lista_SOST.append(elemento)
		distribuzione_di_frequenza = nltk.FreqDist(lista_SOST).most_common(10) # calcolo la distribuzione di frequenza dei bigrammi <AGGETTIVO,SOSTANTIVO>
	for elemento in distribuzione_di_frequenza:
		lista_ordinata.append(elemento[0][0])
	return lista_ordinata

# Funzione per il calcolo della Local Mutual Information e la creazione della lista Sostantivo - Aggetivo - Local Mutual Information
def calcolaLMI (tokens_totali, token_POS, lista_SOST):
	lista_bigrammi = bigrams(token_POS)
	lista_tmp = []
	lista_AGG_SOST = []
	dict_ordinato = {}
	lista_finale = []
	LMI = 0.0
	N = len(tokens_totali)
	for bigramma in lista_bigrammi:
		if bigramma[0][1] in ["JJ", "JJR", "JJS"] and bigramma[1][0] in lista_SOST: # prendo tutti e soli i bigrammi AGG, SOST (dove SOST è uguale agli elementi nella lista dei sostantivi più frequenti
			lista_tmp.append(bigramma)
	lista_AGG_SOST = list(set(lista_tmp)) # elemino le ripetizioni 
	lista_AGG_SOST.sort(key = lambda x: x[1][0], reverse = True) # ordino alfaeticamente in base ai sostantivi, così da avere entrambe le liste ordinate allo stesso modo
	distribuzione_di_frequenza = nltk.FreqDist(lista_AGG_SOST)
	for bigramma in distribuzione_di_frequenza:
		frequenza_SOST = tokens_totali.count(bigramma[1][0]) # frequenza dei soli SOSTANTIVI
		frequenza_AGG = tokens_totali.count(bigramma[0][0]) # frequenza aggettivo
		# frequenza osservata F(a,b)
		frequenza_osservata = distribuzione_di_frequenza[bigramma]
		# # Frequenza attesa F(a) * F(b)/ N
		frequenza_attesa = frequenzaAttesa(frequenza_AGG, frequenza_SOST , N)
		# Local Mutual Infortation LMI = F(a,b) * log2 ( F(a,b) / (F(a)*F(b)/N) ) ==> frequenza osservata * log2 (frequenza osservata / frequenza attesa)
		LMI = (frequenza_osservata*1.0)*math.log((frequenza_osservata*1.0)/(frequenza_attesa*1.0), 2)
		dict_ordinato.setdefault(bigramma[1][0],[]).append([bigramma[0][0],LMI]) #creo dizionario {SOST:[AGG,LMI]}
	# Ciclo per creare la lista finale dal dizionario [SOST,[AGG,LMI]]
	for key, value in dict_ordinato.iteritems():
		value.sort(key = lambda x: x[1], reverse = True) # ordinamento decrescente in base alla LMI
		temp = [key,value]
		lista_finale.append(temp)
	return lista_finale

# Funzione che estrae le entità nominate
def estraiEnitaNominateDiLuoghi(token_POS):
	luoghi = {"GPE":[]} # creo la dict dove inserire solo i nomi di luoghi GPE = geopolitical entity
	chunk = nltk.ne_chunk(token_POS) # applico la Name Entity chunk di nltk per estrapolare informazioni su le entità nominate da tutti i PoS tag e creare l'albero
	# ciclo la lista dei chunk
	for nodo in chunk: 
		lista_NE = ""
		if hasattr(nodo, "label"): # questa condizione verifica se il nodo che stiamo analizzando è un nodo intermedio e non una foglia (la condizione si verifica se troviamo un nodo intermedio)
			if nodo.label() in ["GPE"]:	 # es: nodo.label() ==> "GPE", estrae l'etichetta della NE del nodo.
				for entita in nodo.leaves(): # nodo.leaves() estre la lista di tutte le foglie del nodo intermedio che stiamo analizzando, ciclo questa lista per estrarre il nome del luogo che si trova in posizione [0] della lista
					lista_NE = lista_NE + " " + entita[0] # concateno la foglia (entita[0] == nome del luogo) alla lista NE.
				luoghi[nodo.label()].append(lista_NE) # appendo nella lista, all'interno della dict luoghi, che corrisponde a nodo.label() (in questo caso GPE) la lista delle entità nominate. 
	return luoghi

# Funzione che toglie la punteggiatura
def estraiSenzaPunteggiatura (lista):
	lista_finale = []
	# ciclo che esclude dalla lista gli elementi associati alla punteggiatura
	for elemento in lista:
			if elemento not in [".", ",", ":", ";", "!", "?", "(", ")"]: # controllo della punteggiatura
				lista_finale.append(elemento)
	return lista_finale

# Funzione che estre da una prima lista(lista1) solo gli elementi che si trovano anche nella seconda lista(lista2)
def estraiTipiGrammaticali (lista1, lista2):
	lista_finale = []
	for elemento in lista1:
		if elemento[1] in lista2:
			lista_finale.append(elemento)
	return lista_finale

# Funzione che estrae solo i PoS tag da una lista
def estraiSoloTagPOS (lista):
	lista_finale = []
	for elemento in lista:
		lista_finale.append(elemento[1]) # elemento[1] = POS_tag 
	return lista_finale

# Funzione che conta le occorrenze di una lista
def conta (lista):
	return [(lista.count(elemento), elemento) for elemento in set(lista)]

# Funzine che ordina la lista in ordine decrescente
def ordina (lista):
	return sorted(lista, reverse = True)

# Funzione che divide i nome del fine in input in ["NomeFile", "estensione"]. es: testo.txt => ["testo", "txt"]
def splitterFileName (file):
	nome,estensione = (file.name).split(".") # assegna alla variabile nome il nome del file e alla varibile estensione l'estensione del file 
	return nome

def main(file1, file2):
	file1_input = codecs.open(file1, "r", "utf-8") # apre il "file1", in sola lettura "r", in codifica "utf-8"
	file2_input = codecs.open(file2, "r", "utf-8") # apre il "file2", in sola lettura "r", in codifica "utf-8"
	nome1 = splitterFileName(file1_input) # nome del file senza estensione
	nome2 = splitterFileName(file2_input)
	riga1 = file1_input.read()
	riga2 = file2_input.read()
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') # metodo di lettura del file per la tokenizzazione
	frasi_file1 = sent_tokenizer.tokenize(riga1) # frasi file1
	frasi_file2 = sent_tokenizer.tokenize(riga2) # frasi file2
	lunghezza_corpus_file1, listaToken_file1, POS_token_tag_file1 = CorpusTokensPOS(frasi_file1) # lunghezza del Corpus Token e POS tag
	lunghezza_corpus_file2, listaToken_file2, POS_token_tag_file2 = CorpusTokensPOS(frasi_file2)
	lista_Solo_POS_file1 = estraiSoloTagPOS(POS_token_tag_file1) # estrae solo PoS tag senza token
	lista_Solo_POS_file2 = estraiSoloTagPOS(POS_token_tag_file2)
	token_20_file1, aggettivi_20_file1, verbi_20_file1, POS_10_file1, trigrammi_10_file1 = analisiFrequenze(listaToken_file1, POS_token_tag_file1, lista_Solo_POS_file1) # analisi delle frequenze 
	token_20_file2, aggettivi_20_file2, verbi_20_file2, POS_10_file2, trigrammi_10_file2 = analisiFrequenze(listaToken_file2, POS_token_tag_file2, lista_Solo_POS_file2)
	bigrammi_POS_file1 = bigrams(lista_Solo_POS_file1) # estre coppie (<token-PoS, token-PoS>)
	bigrammi_POS_file2 = bigrams(lista_Solo_POS_file2)
	lista_10_bigrammi_PCong_file1, lista_10_bigrammi_PCond_file1 = probabilita(lista_Solo_POS_file1, bigrammi_POS_file1) # calcola la probabilità
	lista_10_bigrammi_PCong_file2, lista_10_bigrammi_PCond_file2 = probabilita(lista_Solo_POS_file2, bigrammi_POS_file2)
	lista_SOST, lista_SOST_uguali = listaAggSost(POS_token_tag_file1, POS_token_tag_file2)
	LMI_AGG_SOST_1= calcolaLMI(listaToken_file1, POS_token_tag_file1, lista_SOST)
	LMI_AGG_SOST_2= calcolaLMI(listaToken_file2, POS_token_tag_file2, lista_SOST)
	anlisi_linguistica_file1 = estraiEnitaNominateDiLuoghi(POS_token_tag_file1) # analisi delle name entity
	anlisi_linguistica_file2 = estraiEnitaNominateDiLuoghi(POS_token_tag_file2)
	luoghi_10_file1 = nltk.FreqDist(anlisi_linguistica_file1["GPE"]).most_common(20) # estre i 20 nomi di luogo più frequenti
	luoghi_10_file2 = nltk.FreqDist(anlisi_linguistica_file2["GPE"]).most_common(20)
# RISULTATI ANALISI**********************************
# 20 TOKEN
	print "\nIl confronto avviene su due corpus (",nome1,".txt ,",nome2,".txt) i quali contengono: blog scritti da autori di sesso maschile e blog scritti da autori di sesso femminile.\n"
	# LISTA = [Frequenza, token]
	print "\n\n- I 20 TOKEN PIù FREQUENTI (NO PUNTEGGIATURA) -\n"
	print nome1, "\t\t\t\t\t\t",nome2
	for elemento1, elemento2 in zip(token_20_file1, token_20_file2):
		print " Token --> %-20s Freq --> %-20s" % (elemento1[1], elemento1[0]),"Token --> %-20s Freq --> %-20s" % (elemento2[1], elemento2[0])
# AGGETTIVI
	# LISTA = [Frequenza, token]
	print "\n\n- I 20 AGGETTIVI PIù FREQUENTI -\n"
	print nome1, "\t\t\t\t\t\t", nome2
	for elemento1, elemento2 in zip(aggettivi_20_file1, aggettivi_20_file2):
		print " Token --> %-20s Freq --> %-20s" % (elemento1[1][0], elemento1[0]),"Token --> %-20s Freq --> %-20s" % (elemento2[1][0], elemento2[0])
# VERBI
	# LISTA = [Frequenza, token]
	print "\n\n- I 20 VERBI PIù FREQUENTI -\n"
	print nome1, "\t\t\t\t\t\t", nome2
	for elemento1, elemento2 in zip(verbi_20_file1, verbi_20_file2):
		print " Token --> %-20s Freq --> %-20s" % (elemento1[1][0], elemento1[0]),"Token --> %-20s Freq --> %-20s" % (elemento2[1][0], elemento2[0])
# 10 POS
	# LISTA = [Frequenza,POS]
	print "\n\n- I 10 POS TAG PIù FREQUENTI -\n"
	print nome1, "\t\t\t\t\t\t", nome2
	for elemento1, elemento2 in zip(POS_10_file1, POS_10_file2):
		print " PoS --> %-20s Freq --> %-20s" % (elemento1[1], elemento1[0]),"Token --> %-20s Freq --> %-20s" % (elemento2[1], elemento2[0])
# 10 TROGRAMMI DI POS TAG
	# LISTA = [Frequenza,[POS,POS,POS]]
	print "\n\n- I 10 TRIGRAMMI DI POS TAG PIù FREQUENTI -\n"
	print nome1, "\t\t\t\t\t\t\t\t", nome2
	for elemento1, elemento2 in zip(trigrammi_10_file1, trigrammi_10_file2):
		print " Trigramma --> %-3s - %-3s - %-20s Freq --> %-20s" % (elemento1[1][0],elemento1[1][1],elemento1[1][2],elemento1[0])," Trigramma --> %-3s - %-3s - %-20s Freq --> %-20s" % (elemento2[1][0],elemento2[1][1],elemento2[1][2],elemento2[0])
# 10 BIGRAMMI + PROBABILITà
	# LISTA = [Probabilita,[POS,POS]]
	# CONGIUNTA
	print "\n\n- I 10 BIGRAMMI DI POS TAG CON PROBABILITà CONGIUNTA MASSIMA -\n"
	print nome1, "\t\t\t\t\t\t\t\t\t", nome2
	for elemento1, elemento2 in zip(lista_10_bigrammi_PCong_file1, lista_10_bigrammi_PCong_file2):
		print " Bigramma --> %-3s - %-20s  P.Congiunta--> %-0s %-20s" % (elemento1[1][0], elemento1[1][1], "%1.2f" % elemento1[0], "%")," Bigramma --> %-3s - %-20s  P.Congiunta --> %-0s %-20s" % (elemento2[1][0], elemento2[1][1], "%1.2f" % elemento2[0], "%")
	# CONDIZIONATA
	print "\n\n- I 10 BIGRAMMI DI POS TAG CON PROBABILITà CONDIZIONATA MASSIMA -\n"
	print nome1, "\t\t\t\t\t\t\t\t\t", nome2
	for elemento1, elemento2 in zip(lista_10_bigrammi_PCond_file1, lista_10_bigrammi_PCond_file2):
		print " Bigramma --> %-3s - %-20s  P.Condizionata --> %-0s %-20s" % (elemento1[1][0],elemento1[1][1],"%1.2f" % elemento1[0], "%")," Bigramma --> %-3s - %-20s  P.Condizionata  --> %-0s %-20s" % (elemento2[1][0], elemento2[1][1], "%1.2f" % elemento2[0], "%")
#20 SOSTANTIVI AGGETIVI
	# LISTA = [SOST,[AGG,LMI]]
	print "\n\n- I 2O SOSTANTIVI PIù FREQUENTI, E PER OGNIUNO I SUOI AGGETTIVI ORDINATI IN BASE ALLA LOCAL MUTUAL INFORMATION -\n"
	print "\nIL NUMERO DEI SOSTANTIVI TOTALI é 20", "MA NE VERRANO STAMPATI SOLO", len(lista_SOST),"PERCHè", len(lista_SOST_uguali),"SOSTANTIVI SONO PRESENTI NEI 10 SOSTANTIVI PIù FREQUENTI DI ENTRAMBI I CORPUS. QUESTI SOSTANTIVI SONO:"
	for uguale in lista_SOST_uguali:
		print "-", uguale
	print "\n\t\t",nome1,"------------------------------------------------------",nome2
	for lista1, lista2 in zip(LMI_AGG_SOST_1, LMI_AGG_SOST_2):
		print "\n","\t\t\t---------------------------SOST:",lista1[0],"---------------------------"
		for listaValori1, listaValori2 in zip(lista1[1:],lista2[1:]):
			for valore1, valore2 in zip(listaValori1, listaValori2):
				print " \nAGG --> %-10s LMI --> %-10s" % (valore1[0], valore1[1]),"\t\t\t\tAGG --> %-10s LMI --> %-10s" % (valore2[0], valore2[1])	
# 20 NOMI PROPRI DI LUOGO
	print "\n\n- I 20 NOMI PROPRI DI LUOGO PIù FREQUENTI -\n"
	print nome1,":", "\t\t\t\t",nome2,":"
	for elemento1,elemento2 in zip(luoghi_10_file1,luoghi_10_file2):
		print "%-20s Freq --> %-20s" % (elemento1[0], elemento1[1]),"%-20s Freq --> %-20s" % (elemento2[0], elemento2[1]) 
main(sys.argv[1],sys.argv[2])