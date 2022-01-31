#!/usr/bin/env python
import sys, os, random, pygame
from gameVariables import *

def initialize_pygame():
    pygame.init()
    pygame.mixer.init()

    #Opening the window in the center of the screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % ((screenResolution.current_w - gameWidth) / 2, (screenResolution.current_h - gameHeight) / 2)
    screen = pygame.display.set_mode([gameWidth, gameHeight], pygame.DOUBLEBUF, 32)
    pygame.display.set_icon(pygame.image.load('images/icon.ico'))
    pygame.display.set_caption("Flappy Bird")

    return screen

def load_images():
    #- Loading all the images required for the game from the images folder
    #and returning a dictionary of them as following:
       
    #'background_1' : day background
    #'bird' : the bird
    #'bird2' : the bird with the wings down
    #'pipe-up' : the pipe for the upper part
    #'pipe-down' : the pipe for the lower part
    #'ground' : the ground

    def load_image(img_file_name):
        #- Looking for images in the game's images folder (./images/)
        #- Loading the image and then converts it, because it speeds up
        #blitting; returning then the image to the dictionary
        #- For the background image, we load a random one, since we
        #have a day background a night background
        
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background_' + str(random.randint(1, 2)) + '.png'),
            'bird': load_image('bird.png'),
            'bird2': load_image('bird2.png'),    
            'pipe-up': load_image('pipe-up.png'), 
            'pipe-down': load_image('pipe-down.png'),
            'ground': load_image('ground.png')}

def draw_text(screen, text, y_pos, size):
    #Drawing a black text (bigger) and then a white text, smaller
    #over it to get the desired gameScore effect
    font = pygame.font.Font("data/04b_19.TTF", size)
    score_text_b = font.render(str(text), 1, (0, 0, 0))
    score_text_w = font.render(str(text), 1, (255, 255, 255))

    x_pos_b = (gameWidth - score_text_b.get_width()) / 2
    x_pos_w = (gameWidth - score_text_w.get_width()) / 2
    screen.blit(score_text_b, (x_pos_b + 2, y_pos - 1))
    screen.blit(score_text_w, (x_pos_w, y_pos))

def end_the_game(screen, gameScore):
    #Draws a rectangle & shows the gameScore & updates the highscore
    pygame.draw.rect(screen, (0, 0, 0), (23, gameHeight / 2 - 77, 254, 154))
    pygame.draw.rect(screen, (239, 228, 150), (25, gameHeight / 2 - 75, 250, 150))
    draw_text(screen, "Your Score: " + str(gameScore), 200, 35)
    
    f = open("data/highscore", "r+")
    hs = int(f.readline())
    if(gameScore > hs):
       hs = gameScore
       f.seek(0)
       f.truncate()
       f.write(str(hs))
    f.close()
    
    draw_text(screen, "Highscore: " + str(hs), 250, 35)
    draw_text(screen, "Press space to restart", 335, 20)
    draw_text(screen, "Press esc to exit", 355, 20)
    
    #Updates the entire screen for the last time
    pygame.display.update()

    #Gets the keyboard events to se if the user wants to restart the game
    while True:
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == K_SPACE:
                    return 0
                elif e.key == K_ESCAPE:
                    return 1