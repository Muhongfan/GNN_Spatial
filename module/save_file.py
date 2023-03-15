import os
def save_file(dataset, slot, model):
    name = model.__class__.__name__
    # timeslot = 30
    # name = name + str(timeslot)
    path = "./Fusion_"+dataset+"/test_" + str(slot) + '/' + name
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
