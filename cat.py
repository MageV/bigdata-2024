class Cat:
    _already:str=""
    def greet(self,name=''):
        if self._already!=name:
            self._already=name
            print(f"Hello I'm {name}")
        else:
            print("I've already greet you!")


cat=Cat()
cat.greet('cat')
cat.greet('cat')
cat.greet('None')
