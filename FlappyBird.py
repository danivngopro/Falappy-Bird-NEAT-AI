#!/usr/bin/env python

#Importing libraries
import os, sys, pygame, random, math
from pygame.locals import *
from gameFunctions import *
from gameClasses import *
import gameVariables
import neat
from checkPoint import Checkpointer
import time

current_generation = 0

def main(genomes, config):
    #Initializing pygame & mixer
    screen = initialize_pygame()

    #Setting up some timers 
    clock = pygame.time.Clock()
    
    #Loading the images | Creating the bird | Creating the ground | Creating the game list
    nets = []
    birds = []
    gamePipes = []
    gameImages = load_images()
    gameVariables.gameScore = 0
    gameGround = Ground(gameImages['ground'])
      
    #Loading the sounds
    jump_sound = pygame.mixer.Sound('sounds/jump.ogg')
    score_sound = pygame.mixer.Sound('sounds/score.ogg')
    dead_sound = pygame.mixer.Sound('sounds/dead.ogg')

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        birds.append(Bird())

    # Clock Settings
    # Font Settings & Loading Map
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    generationTimer = time.time()
    
    #Loop until...we die!
    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        #Drawing the background
        screen.blit(gameImages['background'], (0, 0))

        # to play yourself disable the code below
        # For Each Car Get The Action It Takes
        for i, bird in enumerate(birds):
            output = nets[i].activate(bird.get_data(gamePipes))
            choice = output.index(max(output))
            if choice == 1:
                # pass
                bird.steps_to_jump = jumpSteps
                # jump_sound.play()
        
        # and enable this code
        # if pygame.mouse.get_pressed()[0]:
        #     for bird in birds:
        #         bird.steps_to_jump = 6
        # else:
        #     print(2)

        # Getting the mouse, keyboard or user events and act accordingly
        if len(gamePipes) == 0:
            p = PipePair(gameWidth, False)
            gamePipes.append(p)
    
        # Check If bird Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, bird in enumerate(birds):
            if bird.is_alive():
                still_alive += 1
                bird.update_position()
                bird.redraw(screen, gameImages['bird'], gameImages['bird2'])
                genomes[i][1].fitness += gameVariables.gameScore * 5

        if still_alive == 0:
            break

        # if time.time() - generationTimer >= 25:
        #     break

        #Tick! (new frame)
        clock.tick(FPS)

        #Updating the position of the gamePipes and redrawing them; if a pipe is not visible anymore,
        #we remove it from the list
        for p in gamePipes:
            p.x -= pixelsFrame
            if p.x <= - pipeWidth:
                gamePipes.remove(p)
            else:
                screen.blit(gameImages['pipe-up'], (p.x, p.toph))
                screen.blit(gameImages['pipe-down'], (p.x, p.bottomh))

        #Redrawing the ground
        gameGround.move_and_redraw(screen)

        #Checks for any collisions between the gamePipes, bird and/or the lower and the
        #upper part of the screen
        for bird in birds:
            if any(p.check_collision((bird.bird_x, bird.bird_y)) for p in gamePipes) or \
                bird.bird_y < 0 or \
                bird.bird_y + birdHeight > gameHeight - groundHeight:
                # dead_sound.play() too caotic
                bird.alive = False

        #There were no collision if we ended up here, so we are checking to see if 
        #the bird went thourgh one half of the pipe's gameWidth; if so, we update the gameScore
        for bird in birds:
            for p in gamePipes:
                if(bird.bird_x > p.x and not p.score_counted):
                    bird.score += 1
                    p.score_counted = True
                    gameVariables.gameScore += 1
                    score_sound.play()

        #Draws the gameScore on the screen
        draw_text(screen, gameVariables.gameScore, 50, 35)

         # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (150, 15)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (150, 40)
        screen.blit(text, text_rect)

        #Updates the screen
        pygame.display.update()

if __name__ == '__main__':
    checkpointer = Checkpointer()
    try:
        # if possible, load a pre-trained nn
        checkpoint = checkpointer.restore_checkpoint('./neat-checkpoint-50')
        config = checkpoint[0]
        population = checkpoint[1]
        current_generation = checkpoint[3]

        population.run(main, 50)

    except:
        # Load Config
        config_path = "./config.txt"
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    config_path)

        # Create Population And Add Reporters
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Run Simulation For A Maximum of 1000 Generations
        population.run(main, 50)

    # save the nn
    checkpointer.save_checkpoint(config, population, neat.DefaultSpeciesSet , current_generation)
