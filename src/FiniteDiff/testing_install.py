import matplotlib.pyplot as plt
import ipywidgets as widgets
class MyClass:
    def __init__(self,val) -> None:
        self.val = val
    def __str__(self) -> str:
        return f"a cute little object with value {self.val}"
    
    def interact(self):

        fig, ax = plt.subplots(figsize=(6, 4))

        floater = widgets.BoundedFloatText(value = 3,min = 0,max = 10)




        @widgets.interact(Time = floater)
        def update(Time=0):
            for artist in plt.gca().lines + plt.gca().collections:
                artist.remove()
            
            ax.set_ylim([0,10])
            ax.plot([0,1],[Time,Time])
            
            ax.set_ylabel("just moving :)")

