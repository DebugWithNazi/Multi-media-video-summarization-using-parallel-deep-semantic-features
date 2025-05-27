import tkinter as tk

window_main = tk.Tk(className='Tkinter - TutorialKart')
window_main.geometry("400x200")

frame_1 = tk.Frame(window_main, bg='#c4ffd2', width=200, height=50)
frame_1.pack()
frame_1.pack_propagate(0)

frame_2 = tk.Frame(window_main, bg='#ffffff', width=350, height=70)
frame_2.pack()
frame_2.pack_propagate(0)

# in frame_1
label_1 = tk.Label(frame_1, text='Name')
label_1.pack(side=tk.LEFT)
submit = tk.Entry(frame_1)
submit.pack(side=tk.RIGHT)

# in frame_2
button_reset = tk.Button(frame_2, text='Reset')
button_reset.pack(side=tk.LEFT)
button_submit = tk.Button(frame_2, text='Submit')
button_submit.pack(side=tk.RIGHT)

window_main.mainloop()