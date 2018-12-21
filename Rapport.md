# final-lab-GourramSoumia
final-lab-GourramSoumia created by GitHub Classroom

## Classification des documents du procès des groupes américains du tabac

### Introduction
Le gouvernement américain a attaqué en justice cinq grands groupes américains du tabac pour avoir amassé d'importants bénéfices en mentant sur les dangers de la cigarette. Le cigarettiers  se sont entendus dès 1953, pour "mener ensemble une vaste campagne de relations publiques afin de contrer les preuves de plus en plus manifestes d'un lien entre la consommation de tabac et des maladies graves".
Dans ce procès 14 millions de documents ont été collectés et numérisés. Afin de faciliter l'exploitation de ces documents par les avocats, vous êtes en charge de mettre en place une classification automatique des types de documents.
Un échantillon aléatoire des documents a été collecté et des opérateurs ont classé les documents dans des répertoires correspondant aux classes de documents : lettres, rapports, notes, email, etc.


### Analyse de données 
```ruby
data = pd.read_csv("Tobacco3482.csv")
print (" Le nombre d'article ", len(data))
```
Notre base de données contient 3482 articles, Contenant 10 classes qui sont répartie comme ceci : 

![countplot](https://user-images.githubusercontent.com/44871503/50358534-f0b2f900-0559-11e9-9674-4f1478ebcf2c.png)


### Classification de données

Avant de commencer la classification, on divise notre base de données en 3 partie ( apprentissage, test et validation). La base d'apprentissage prend 80% de la base de données, la base de test prend 15% et la base de validation 5%.
````ruby
X_train,X_test, y_train,y_test = train_test_split(DF.text, DF.label, test_size=0.20, 
                                                random_state=1)

X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.20, 
                                                random_state=1)

print('nb apprentissage :' ,X_train.shape)
print('nb test:', X_test.shape)
print('nb validation:', X_dev.shape)

````

#### Naive Bayes
Le premier classifieur choisie c'est la naive bayes.
````ruby
model = MultinomialNB()
model.fit(X_train_vect, y_train)
````
On obtiens les performances suivantes :
````ruby
Le score obtenu pour les données d app: 84.29084380610414 %
Le score obtenu pour les données de test: 72.02295552367288 %
Le score obtenu pour les données de val: 73.24955116696589 %
````
On remarque que avec une représentation Tf-Idf on obtiens des performances moins.
````
Le score obtenu pour les données d app: 75.76301615798923 %
Le score obtenu pour les données de test: 66.42754662840747 %
Le score obtenu pour les données de val: 67.3249551166966 %
````
La matrice de covariance:

![matrice_cov](https://user-images.githubusercontent.com/44871503/50359557-7ab09100-055d-11e9-8875-faa2b2de43ee.png)

#### MLP
Concernant la classification avec un MLP on obtient des meilleurs résultats.
````ruby
classifier= MLPClassifier(alpha = 1)
classifier.fit(X_train_vect, y_train)
````
````ruby
Le score obtenu pour les données d app: 98.33931777378815 %
Le score obtenu pour les données de test: 78.33572453371592 %
Le score obtenu pour les données de val: 78.63554757630162 %
````







