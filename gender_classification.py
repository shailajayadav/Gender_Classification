from sklearn import tree
#[height,weight,shoe_size]
data_x=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
data_Y=['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
cls=tree.DecisionTreeClassifier()
cls=cls.fit(data_x,data_Y)

prediction=cls.predict([[181, 80, 44]])
print(prediction)