import basic_hopfield_network as hn

patrones_memoria = [[0,0,1,1], [1,1,0,0], [0,1,0,1]]
patron_prueba = [0, 0, 0, 1]

red = hn.HopfieldNetwork(4)
red.train(patrones_memoria)
print(red.synchronous_update(patron_prueba))
print(red.asynchronous_update(patron_prueba))
