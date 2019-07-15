# -*- coding: utf-8 -*-
import sys, nltk, codecs

# funzione per definire la lunghezza del corpus e la lista di tutti i token
def CorpusTokensVocabolario (frasi):
	lista_tokens = []
	tot_frasi = 0
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		lista_tokens = lista_tokens + tokens
		tot_frasi = tot_frasi + 1
	corpus = len(lista_tokens)
	vocabolario = set(lista_tokens)
	return corpus, lista_tokens, vocabolario, tot_frasi

# funzione per il coneggio di frasi, caratteri
def contaFrasiECaratteri(frasi, listaToken): 
	#dichiaro contatori a 0
	conta_frasi = 0
	conta_caratteri = 0
	for frase in frasi:
		conta_frasi = conta_frasi + 1 # incemento conte_frasi di uno ogni volta che eseguo il ciclo for
	for token in listaToken:
		conta_caratteri = conta_caratteri + len(token) # incrementa numerico in base al numero del token
	return conta_frasi, conta_caratteri

# funzione che restituisce una lista contente l'incremento del vocabolario ogni 1000 token, l'incremento degli hapax ogni 1000 token, e la Type Token Ratio
def vocabolarioHapaxTTR1000 (token_totali):
	lista_finale_vocabolario = [] 
	lista_finale_hapax = []
 	lista_incrementale = [] # lista vuota per concatenare le liste ogni 1000 token letti
 	lista_vocabolario = [] # lista con solo il vocabolario della lista incrementale
 	TTR = 0.0
	lista_composta = [token_totali[x:x+1000] for x in range(0, len(token_totali),1000)] # split della lista dei token totali creando sotto liste ogni 1000 token
	for lista in lista_composta: # itero la lista composta per ogni sotto lista
		lista_incrementale = lista_incrementale + lista # concateno la lista incremento con la lista della lista_composta
		lista_vocabolario = set(lista_incrementale) # calcolo il vocabolario ogni volta che aggiungo 1000 token alla lista incremento
		contatore = len(lista_vocabolario) # conto il numero di token del vocabolario
		lista_finale_vocabolario.append(contatore) # appendo i numero trovato nella posizione i-esima nel vettore finale vocabolario
		numero_hapax = hapax1000(lista_incrementale, lista_vocabolario) # richiamo la funzione per il conteggio deglio hapax
		lista_finale_hapax.append(numero_hapax)# appendo il numero trovato nella posizione i-esima nel vettore finale vocabolario
		if len(lista_incrementale) == 5000: # raggiunti i 5000 token calcolo la Type Token Ratio
			corpus = float(len(lista_incrementale)) # converto in float la lunghezza della lista incrementale
			vocabolario = float(len(lista_vocabolario))  # converto in float la lunghezza della lista vocabolario
			TTR = vocabolario/corpus # formula type token ration (TTR = Vc/|C|)
	return lista_finale_vocabolario, lista_finale_hapax, TTR

# funzione che conta gli hapax
def hapax1000 (token_totali,vocabolario):
	conta = 0
	for tok in vocabolario:
		frequenza_token = token_totali.count(tok) # calcolo la frequenza di quel token (tok) in quel corpus (token_totali)
		if frequenza_token == 1: # se la frequenza del token è 1 è un hapax 
			conta = conta + 1 #conto quanti hapax trovo
	return conta

# funzione che definisce le percentuali di SOSTANTIVI AGGETTIVI VERBI e PRONOMI nel testo e il loro numero medio nelle frasi
def analisiLinguistica (token_totali, frasi_totali):
	lista_token_e_tag = nltk.pos_tag(token_totali) # analizzo il testo creando una lista i cui elementi sono bigrammi contenenti le coppie <token, POS>
	corpus = float(len(token_totali))
	lista_POS = []
	SOSTANTIVI = float(0)
	AGGETTIVI = float(0)
	VERBI = float(0)
	PRONOMI = float(0)
	for bigramma in lista_token_e_tag: # per ogni bigramma che sta nella lista_token_e tag
		lista_POS.append(bigramma[1]) # creo una lista contenente solo e POS dei bigrammi
	for tipo in lista_POS: # leggo la lista e confronto i valori con quelli che sto cercando
		if tipo in ["NN", "NNS", "NNP", "NNPS"]: # SOSTANTIVI
			SOSTANTIVI =SOSTANTIVI + 1
		if tipo in ["JJ", "JJR", "JJS"]: # AGGETTIVI
			AGGETTIVI = AGGETTIVI + 1
		if tipo in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]: # VERBI
			VERBI = VERBI + 1
		if tipo in ["PRP", "PRP$", "WP$", "WP"]: # PRONOMI
			PRONOMI = PRONOMI + 1
	# PERCENTUALI
	percentuale_SOSTANTIVI = (SOSTANTIVI/corpus)*100
	percentuale_AGGETTIVI = (AGGETTIVI/corpus)*100
	percentuale_VERBI = (VERBI/corpus)*100
	percentuale_PRONOMI = (PRONOMI/corpus)*100
	# MEDIE PER FRASE
	media_SOSTANTIVI = (SOSTANTIVI/frasi_totali)
	media_AGGETTIVI = (AGGETTIVI/frasi_totali)
	media_VERBI = (VERBI/frasi_totali)
	media_PRONOMI = (PRONOMI/frasi_totali)
	return percentuale_SOSTANTIVI, percentuale_AGGETTIVI, percentuale_VERBI, percentuale_PRONOMI, media_SOSTANTIVI, media_AGGETTIVI, media_VERBI, media_PRONOMI

def splitterName (file):
	nome,estensione = (file.name).split(".") # assegan alla variabile nome il nome del file e alla varibile estensione l'estensione del file 
	return nome

#funzione principale
def main(file1, file2):
	file1_input = codecs.open(file1, "r", "utf-8") # apre il "file1", in sola lettura "r", in codifica "utf-8"
	file2_input = codecs.open(file2, "r", "utf-8") # apre il "file2", in sola lettura "r", in codifica "utf-8"
	nome1 = splitterName(file1_input) # nome del file senza estensione
	nome2 = splitterName(file2_input)
	riga1 = file1_input.read()
	riga2 = file2_input.read()
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') # metodo di lettura del file per la tokenizzazione
	frasi_file1 = sent_tokenizer.tokenize(riga1) # frasi file1
	frasi_file2 = sent_tokenizer.tokenize(riga2) # frasi file2
	lughezza_corpus_file1, listaToken_file1, vocabolario_file1, tot_frasi_file1 = CorpusTokensVocabolario(frasi_file1) # Lunghezza del Corpus Token e Vocabolario
	lughezza_corpus_file2, listaToken_file2, vocabolario_file2, tot_frasi_file2 = CorpusTokensVocabolario(frasi_file2)
	tot_token_file1 = len(listaToken_file1)
	tot_token_file2 = len(listaToken_file2)
	numero_tot_frasi_file1, tot_caratteri_file1 = contaFrasiECaratteri(frasi_file1, listaToken_file1) # richiamo funzione conta() per conteggio frasi e caratteri
	numero_tot_frasi_file2, tot_caratteri_file2 = contaFrasiECaratteri(frasi_file2, listaToken_file2)
	lunghezza_media_frasi_file1 = tot_token_file1/numero_tot_frasi_file1 # lunghezza media delle frasi in termini di token
	lunghezza_media_frasi_file2 = tot_token_file2/numero_tot_frasi_file2
	lunghezza_media_parole_file1 = tot_caratteri_file1/tot_token_file1 # lunghezza media delle parole in termini di caratteri
	lunghezza_media_parole_file2 = tot_caratteri_file2/tot_token_file2
	vocabolario_1000_file1, hapax_1000_file1, Type_Token_Ratio_5000_file1 = vocabolarioHapaxTTR1000(listaToken_file1) # vocabolario hapax ogni 1000 token, type token ration raggiunti i 5000
	vocabolario_1000_file2, hapax_1000_file2, Type_Token_Ratio_5000_file2 = vocabolarioHapaxTTR1000(listaToken_file2)
	SOSTANTIVI_file1, AGGETTIVI_file1, VERBI_file1, PRONOMI_file1,media_SOSTANTIVI_file1, media_AGGETTIVI_file1, media_VERBI_file1, media_PRONOMI_file1 = analisiLinguistica(listaToken_file1, tot_frasi_file1) # analisi linguistica che restituise percentuale tot e media per frase  
	SOSTANTIVI_file2, AGGETTIVI_file2, VERBI_file2, PRONOMI_file2, media_SOSTANTIVI_file2, media_AGGETTIVI_file2, media_VERBI_file2, media_PRONOMI_file2 = analisiLinguistica(listaToken_file2, tot_frasi_file2)
	# RISULTATI **************************************************
	print "\nIl confronto avviene su due corpus (",nome1,".txt ,",nome2,".txt) i quali contengono: blog scritti da autori di sesso maschile e blog scritti da autori di sesso femminile.\n"
	print "- CONFRONTI BASILARI -"
	print"\n",nome1,"\t\t\t",nome2,"\n"
	# TOT
	print "Frasi file1 -->",numero_tot_frasi_file1, "\t\tFrasi file2 -->", numero_tot_frasi_file2 # stampa il totale delle frasi dei due file
	print "Token file1 -->", tot_token_file1,"\t\tToken file2", tot_token_file2 # stampa il totale dei token dei due file
	print "Media frasi file1 -->",lunghezza_media_frasi_file1,"\tMedia frasi file2 -->",lunghezza_media_frasi_file2 # stampa la lunghezza media delle frasi in base ai token
	print "Media parole file1 -->",lunghezza_media_parole_file1,"\tMedia parole file2 -->",lunghezza_media_parole_file2 # stampa la lunghezza media delle parole in base ai caratteri
	# confronto numero frasi
	if numero_tot_frasi_file1 > numero_tot_frasi_file2:
		print "\n\n- Il numero totale delle frasi scritte nei", nome1, "è maggiore di quelle scritte nei", nome2
	elif numero_tot_frasi_file1 == numero_tot_frasi_file2:
		print "- Il numero totale delle frasi scritte nei", nome1, "e di quelle scritte nei", nome2, ", è lo stesso"
	else:
		print "- Il numero totale delle frasi scritte nei", nome1, "è maggiore a quelle scritte nei", nome2
	# confronto token totali
	if tot_token_file1 > tot_token_file2:
		print "- I",nome1,"usano un numero di token maggiore rispetto ai ", nome2
	elif tot_token_file1 == tot_token_file2:
		print "- I token usati da entarbi i corpus è uguale"
	else:
		print "- I", nome2,"usamo un numero di token maggiore rispetto ai ", nome1
	# confronto media frasi in termini di token
	if lunghezza_media_frasi_file1 > lunghezza_media_frasi_file2:
		print "- I", nome1, "tendono ad avere una lunghezza media delle frasi, in termini di token, più ampia rispetto ai", nome2
	elif lunghezza_media_frasi_file1 == lunghezza_media_frasi_file2:
		print "- La lunghezza media delle frasi in termini di token di entrambi i corpus è uguale"
	else:
		print "- I", nome2, "tendono ad avere una lunghezza media delle frasi, in termini di token, più ampia rispetto ai", nome1
	# confronto media parole in termine di caratteri
	if lunghezza_media_parole_file1 > lunghezza_media_parole_file2:
		print "- I", nome1, "tendono ad avere una media della lunghezza media delle parole, in termini di token, più ampia rispetto ai", nome2
	elif lunghezza_media_parole_file1 == lunghezza_media_parole_file2:
		print "- Lunghezza media di parole in termini di caratteri di entrambi i corpus è uguale"
	else:
		print "- I", nome2, "tendono ad avere una media della lunghezza media delle parole, in termini di token, più ampia rispetto ai", nome1
	# VOCABOLARIO
	print "\n\n- INCREMENTO DEL VOCABOLARIO OGNI 1000 TOKEN -\n" 
	print nome1, "\t\t",nome2
	for e1, e2 in zip(vocabolario_1000_file1, vocabolario_1000_file2):
		print " - %-20s" % (e1),"- %-20s" % (e2)
	# HAPAX
	print "\n\n- INCREMENTO DEGLI HAPAX OGNI 1000 TOKEN - \n"
	print nome1, "\t\t",nome2
	for e1, e2 in zip(hapax_1000_file1, hapax_1000_file2):
		print " - %-20s" % (e1),"- %-20s" % (e2)
	#Type Token Ration (TTR)
	print "\n\n- TYPE TOKEN RATIO PRIMI 5000 TOKEN -\n" 
	print  nome1," -->   ", "%1.2f" % Type_Token_Ratio_5000_file1, "\n",nome2," -->   ", "%1.2f" % Type_Token_Ratio_5000_file2, 
	# DISTRIBUZIONE % di SOSTANTIVI AGGETTIVI VERBI E PRONOMI
	print "\n\n- DISTRIBUZIONE PERCENTUALE SOSTANTIVI AGGETTIVI VERBI E PRONOMI -\n"
	print nome1," -->   SOSTANTIVI = ", "%1.2f" % SOSTANTIVI_file1,"%","\t AGGETTIVI = ", "%1.2f" % AGGETTIVI_file1,"%","\t VERBI = ", "%1.2f" % VERBI_file1,"%","\t PRONOMI = ", "%1.2f" % PRONOMI_file1,"%"
	print nome2," -->   SOSTANTIVI = ", "%1.2f" % SOSTANTIVI_file2,"%","\t AGGETTIVI = ", "%1.2f" % AGGETTIVI_file2,"%","\t VERBI = ", "%1.2f" % VERBI_file2,"%","\t PRONOMI = ", "%1.2f" % PRONOMI_file2,"%"
	# MEDIA PER FRASE DI SOSTANTIVI AGGETTIVI VERBI E PRONOMI
	print "\n\n- MEDIA PER FRASE DEI SOSTANTIVI AGGETTIVI VERBI E PRONOMI -\n"
	print nome1," -->   SOSTANTIVI = ", "%1.0f" % media_SOSTANTIVI_file1,"\t AGGETTIVI = ", "%1.0f" % media_AGGETTIVI_file1,"\t VERBI = ", "%1.0f" % media_VERBI_file1,"\t PRONOMI = ", "%1.0f" % media_PRONOMI_file1
	print nome2," -->   SOSTANTIVI = ", "%1.0f" % media_SOSTANTIVI_file2,"\t AGGETTIVI = ", "%1.0f" % media_AGGETTIVI_file2,"\t VERBI = ", "%1.0f" % media_VERBI_file2,"\t PRONOMI = ", "%1.0f" % media_PRONOMI_file2
main(sys.argv[1],sys.argv[2])