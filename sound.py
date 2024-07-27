import pygame
import time
import os
def alarm(flag):
    """
    Plays an MP3 file repeatedly until the flag becomes False.
    """
    while flag:
        pygame.mixer.init()
        pygame.mixer.music.load(os.getcwd()+"/sound.mp3")
        pygame.mixer.music.play()
        time.sleep(1)

    pygame.mixer.music.stop()
    
    pygame.mixer.quit()
alarm(True)