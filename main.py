import videoTester
from tkinter import *
from tkinter import filedialog
import pygame
import time
# to determine the length of song
from mutagen.mp3 import MP3





# grab song length time info
def play_time():
    # grab time
    current_time = pygame.mixer.music.get_pos() / 1000
    converted_time = time.strftime('%M:%S',time.gmtime(current_time))


    # grabbing current song tuple number
    #current_song = song_box.curselection()

    # grab song title from playlist
    song = song_box.get(ACTIVE)
    # renaming the song
    song = f'D:/Realtime-Emotion-Detection-master/songs/{song}'

    # get song length
    song_mut = MP3(song)
    song_length = song_mut.info.length
    # convert the song length to time format
    converted_song_length = time.strftime('%M:%S', time.gmtime(song_length))



    # output time to status bar
    status_bar.config(text = f'Time Elapsed:{converted_time}  of  {converted_song_length}')
    # update time
    status_bar.after(1000,play_time)



# add song
def add_song():
    emotion = videoTester.emotionCapture()
    song = filedialog.askopenfilename(initialdir=f'songs/{emotion}',title = "Choose A Song",filetypes=(("mp3 Files","*.mp3"),))
    # strip the extension
    song = song.replace(f"D:/Realtime-Emotion-Detection-master/songs/","")

    # add song to listbox
    song_box.insert(END,song)

# add man song to paylist
def add_many_song():
    emotion = videoTester.emotionCapture()
    songs = filedialog.askopenfilenames(initialdir=f'songs/{emotion}', title="Choose A Song",filetypes=(("mp3 Files", "*.mp3"),))

    # loop through list
    for song in songs:
        # strip the extension
        song = song.replace(f"D:/Realtime-Emotion-Detection-master/songs/", "")
        # add song to listbox
        song_box.insert(END, song)


# play selected song
def play():
    # get the active song
    song = song_box.get(ACTIVE)
    song = f'D:/Realtime-Emotion-Detection-master/songs/{song}'
    pygame.mixer.music.load(song)
    pygame.mixer.music.play(loops=0)
    # call the play time fuction to get song length
    play_time()



# stop playing current song
def stop():
    pygame.mixer.music.stop()
    song_box.selection_clear(ACTIVE)


#play the next song in playlist
def next_song():
    #grabbing current song tuple number
    next_one = song_box.curselection()
    # add 1 to current song number
    next_one = next_one[0]+1
    #grab song title from playlist
    song = song_box.get(next_one)
    #renaming the song
    song = f'D:/Realtime-Emotion-Detection-master/songs/{song}'
    pygame.mixer.music.load(song)
    pygame.mixer.music.play(loops=0)

    # clear active bar in playlist listbox
    song_box.selection_clear(0,END)
    # activate new song bar
    song_box.activate(next_one)
    # set active ba to next song
    song_box.selection_set(next_one,last=None)

#play previous_song

def previous_song():
    # grabbing current song tuple number
    next_one = song_box.curselection()
    # add 1 to current song number
    next_one = next_one[0] - 1
    # grab song title from playlist
    song = song_box.get(next_one)
    # renaming the song
    song = f'D:/Realtime-Emotion-Detection-master/songs/{song}'
    pygame.mixer.music.load(song)
    pygame.mixer.music.play(loops=0)

    # clear active bar in playlist listbox
    song_box.selection_clear(0, END)
    # activate new song bar
    song_box.activate(next_one)
    # set active ba to next song
    song_box.selection_set(next_one, last=None)


# delete a song
def delete_song():
    #delete active song
    song_box.delete(ANCHOR)
    pygame.mixer.music.stop()

# delete all song
def delete_all_songs():
    #delete all songs in listbox
    song_box.delete(0,END)
    #stops the music if playing
    pygame.mixer.music.stop()








# global pause variable
global paused
paused = False

#pause and unpause current song
def pause(is_paused):
    global paused
    paused = is_paused
    if paused:
        # pause music
        pygame.mixer.music.unpause()
        paused = False
    else:
        # unpause music
        pygame.mixer.music.pause()
        paused = True






root = Tk()
root.title("MP3 Player")
root.geometry("500x300")

#emotion = videoTester.emotionCapture()
#print(emotion)
# intitialize pygame
pygame.mixer.init()

song_box = Listbox(root,bg = "black",fg = "green",width = 60,selectbackground ="gray",selectforeground = "black")
song_box.pack(pady = 20)



# player control image
back_btn_img = PhotoImage(file = 'images/prev50.png')
forward_btn_img = PhotoImage(file = 'images/next50.png')
play_btn_img = PhotoImage(file = 'images/play50.png')
pause_btn_img = PhotoImage(file = 'images/pause50.png')
stop_btn_img = PhotoImage(file = 'images/stop50.png')


# create player control frame
controls_frame = Frame(root)
controls_frame.pack()

# button creation
back_button = Button(controls_frame,image = back_btn_img,borderwidth = 0,command = previous_song)
play_button = Button(controls_frame,image = play_btn_img,borderwidth = 0,command = play)
forward_button = Button(controls_frame,image = forward_btn_img,borderwidth = 0,command = next_song)
pause_button = Button(controls_frame,image = pause_btn_img,borderwidth = 0,command = lambda: pause(paused))
stop_button = Button(controls_frame,image = stop_btn_img,borderwidth = 0,command = stop)

back_button.grid(row=0 ,column=0)
pause_button.grid(row=0,column=1)
play_button.grid(row=0,column=2)
stop_button.grid(row=0,column=3)
forward_button.grid(row=0,column=4)


# create menu
my_menu = Menu(root)
root.config(menu = my_menu)

# Add one song to menu
add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Add Song",menu = add_song_menu)
add_song_menu.add_command(label = "Add one song to playlist",command = add_song)

#add many song to playlist
add_song_menu.add_command(label = "Add many song to playlist",command = add_many_song)

# create Delete song Menu
remove_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Remove Song",menu = remove_song_menu)
remove_song_menu.add_command(label="Delete A song from Playlist", command = delete_song)
remove_song_menu.add_command(label="Delete All song from Playlist",command = delete_all_songs)


# create status bar
status_bar = Label(root,text="",bd=1,relief=GROOVE,anchor=E)#east
status_bar.pack(fill=X,side=BOTTOM,ipady=2)


root.mainloop()