# Universidad del Valle de Guatemala
# Graficas por computadora
# Maria Fernanda Estrada 14198
# Proyecto model viewer
# 18/05/2019

import random
import numpy
import glm
import pyassimp
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from math import *



# Shaders
vertex_shader = """
#version 330
layout (location = 0) in vec4 position;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec2 texcoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec4 color;
uniform vec4 light;

out vec4 vertexColor;
out vec2 vertexTexcoords;

void main()
{
    float intensity = dot(normal, normalize(light - position));

    gl_Position = projection * view * model * position;

    if (color[0] == 155 && color[1] == 155 && color[2] == 155){
        vertexColor = color;
    }else{
        vertexColor = color * intensity;
    }
    vertexTexcoords = texcoords;
}
"""

# Shaders
fragment_shader = """
#version 330

layout (location = 0) out vec4 diffuseColor;

in vec4 vertexColor;
in vec2 vertexTexcoords;

uniform sampler2D tex;

void main()
{
    vec4 textureColor = texture(tex, vertexTexcoords);
    if (vertexColor[0] == 155 && vertexColor[1] == 155 && vertexColor[2] == 155){
        diffuseColor[0] = textureColor[2];
        diffuseColor[1] = textureColor[0];
        diffuseColor[2] = textureColor[1];
        diffuseColor[3] = 255;
    }else{
        diffuseColor = vertexColor * textureColor;
    }
}
"""


# Variable para invertir el color del modelo
invert = False

# Leer y mostrar modelo
def glize(node):
    model = node.transformation.astype(numpy.float32)

    for mesh in node.meshes:
        material = dict(mesh.material.properties.items())
        texture = material['file']

        texture_surface = pygame.image.load(texture)
        texture_data = pygame.image.tostring(texture_surface, "RGB", 1)
        width = texture_surface.get_width()
        height = texture_surface.get_height()

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        vertex_data = numpy.hstack((
            numpy.array(mesh.vertices, dtype=numpy.float32),
            numpy.array(mesh.normals, dtype=numpy.float32),
            numpy.array(mesh.texturecoords[0], dtype=numpy.float32)
        ))

        index_data = numpy.hstack((
            numpy.array(mesh.faces, dtype=numpy.int32)
        ))

        vertex_buffer_object = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, False, 9*4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 9*4, ctypes.c_void_p(3*4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, False, 9*4, ctypes.c_void_p(6*4))
        glEnableVertexAttribArray(2)


        element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)

        glUniformMatrix4fv(
            glGetUniformLocation(shader, "model"), 1, GL_FALSE, model
        )

        glUniformMatrix4fv(
            glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view)
        )

        glUniformMatrix4fv(
            glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection)
        )
        
        global invert

        diffuse = mesh.material.properties['diffuse']

        if invert:
            diffuse = (155,155,155)

        glUniform4f(
            glGetUniformLocation(shader, "color"),
            *diffuse,
            1
        )

        glUniform4f(
            glGetUniformLocation(shader, "light"),
            4, 30, 4, 20
        )

        glDrawElements(GL_TRIANGLES, len(index_data), GL_UNSIGNED_INT, None)

    for child in node.children:
        glize(child)

# Configuraciones
clock = pygame.time.Clock()

screen = pygame.display.set_mode((800, 600), pygame.OPENGL|pygame.DOUBLEBUF)

glClearColor(0.6,0.6,0.6,1.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D)

shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

glUseProgram(shader)

model = glm.mat4(1)
view = glm.mat4(1)
projection = glm.perspective(glm.radians(45), 800/600, 0.1, 1000.0)

glViewport(0, 0, 800, 600)

camera = glm.vec3(20, 0, 10)

scene = pyassimp.load('OedoCastle.obj')
    

while True:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    view = glm.lookAt(camera, glm.vec3(0,0,0), glm.vec3(0, 1, 0))

    glize(scene.rootnode)

    pygame.display.flip()

    # Para las teclas de control
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                angulo = atan2(camera.x,camera.z) - 0.1
                distancia = ((camera.x**2)+(camera.z**2))**0.5
                camera.z = cos(angulo) * distancia
                camera.x = sin(angulo) * distancia

            elif event.key == pygame.K_RIGHT:
                angulo = atan2(camera.x,camera.z) + 0.1
                distancia = ((camera.x**2)+(camera.z**2))**0.5
                camera.z = cos(angulo) * distancia
                camera.x = sin(angulo) * distancia

            elif event.key == pygame.K_DOWN:
                if camera.y > 2:
                    camera.y -= 1

            elif event.key == pygame.K_UP:
                if camera.y < 12:
                    camera.y += 1

            elif event.key == pygame.K_z:
                angulo = atan2(camera.x,camera.z)
                distancia = ((camera.x**2)+(camera.z**2))**0.5 - 1
                if distancia > 15:
                    camera.z = cos(angulo) * distancia
                    camera.x = sin(angulo) * distancia
            
            elif event.key == pygame.K_x:
                angulo = atan2(camera.x,camera.z)
                distancia = ((camera.x**2)+(camera.z**2))**0.5 + 1
                if distancia < 25:
                    camera.z = cos(angulo) * distancia
                    camera.x = sin(angulo) * distancia
            elif event.key == pygame.K_c:
                if invert:
                    invert = False
                else:
                    invert = True


    clock.tick(15)