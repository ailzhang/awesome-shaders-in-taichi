# https://www.shadertoy.com/view/XslGRr#
import time
import taichi as ti
import taichi.math as tm
from PIL import Image

ti.init(arch=ti.vulkan)

W, H = 960, 640

iResolution = tm.vec2(W, H)
iTime = ti.field(float, shape=())
iMouse = ti.Vector.field(2, float, shape=())

img = ti.Vector.field(3, float, shape=(W, H))

sundir = tm.vec3(-0.7071, 0., -0.7071)

def init():
    iTime[None] = 0.
    iMouse[None] = [0., 0.]

@ti.func
def setCamera(ro: ti.template(), ta: ti.template(), cr: ti.f32):
    cw = tm.normalize(ta - ro)
    cp = tm.vec3(tm.sin(cr), tm.cos(cr), 0.)
    cu = tm.normalize(tm.cross(cw, cp))
    cv = tm.normalize(tm.cross(cu, cw))
    return tm.mat3(cu, cv, cw)

@ti.func
def noise(x: ti.template(), i0: ti.template()):
    p = tm.floor(x)
    f = x - p # tm.frac(x)
    f = f * f * (3. - 2. * f)
    uv = (p.xy + tm.vec2(37., 239.) * p.z) + f.xy
    rg = i0.sample_lod((uv + 0.5) / 256., 0).yx
    return tm.mix(rg.x, rg.y, f.z) * 2.0 - 1.0

@ti.func
def map5(p: ti.template(), i0: ti.template()):
    q = p - tm.vec3(0., 0.1, 1.0) * iTime[None]
    f = 0.5 * noise(q, i0)
    q = q * 2.02
    f += 0.25 * noise(q, i0)
    q = q * 2.03
    f += 0.125 * noise(q, i0)
    q = q * 2.01
    f += 0.06250 * noise(q, i0)
    q = q * 2.02
    f += 0.03125 * noise(q, i0)
    return tm.clamp(1.5 - p.y - 2.0 + 1.75 * f, 0., 1.)

@ti.func
def map4(p: ti.template(), i0: ti.template()):
    q = p - tm.vec3(0., 0.1, 1.0) * iTime[None]
    f = 0.5 * noise(q, i0)
    q = q * 2.02
    f += 0.25 * noise(q, i0)
    q = q * 2.03
    f += 0.125 * noise(q, i0)
    q = q * 2.01
    f += 0.06250 * noise(q, i0)
    return tm.clamp(1.5 - p.y - 2.0 + 1.75 * f, 0., 1.)

@ti.func
def map3(p: ti.template(), i0: ti.template()):
    q = p - tm.vec3(0., 0.1, 1.0) * iTime[None]
    f = 0.5 * noise(q, i0)
    q = q * 2.02
    f += 0.25 * noise(q, i0)
    q = q * 2.03
    f += 0.125 * noise(q, i0)
    return tm.clamp(1.5 - p.y - 2.0 + 1.75 * f, 0., 1.)

@ti.func
def map2(p: ti.template(), i0: ti.template()):
    q = p - tm.vec3(0., 0.1, 1.0) * iTime[None]
    f = 0.5 * noise(q, i0)
    q = q * 2.02
    f += 0.25 * noise(q, i0)
    return tm.clamp(1.5 - p.y - 2.0 + 1.75 * f, 0., 1.)

@ti.func
def raymarch(ro: ti.template(), rd: ti.template(), bgcol: ti.template(), px: ti.template(), i0: ti.template(), i1: ti.template()):
    sum = tm.vec4(0.)
    t = 0.05 * i1.fetch(px & 255, 0).x
    for i in range(40):
        pos = ro + t * rd
        if pos.y < -3. or pos.y >2. or sum.a >0.99:
            break
        den = map5(pos, i0)
        if den > 0.01:
            dif = tm.clamp((den - map5(pos + 0.3 * sundir, i0)) / 0.6, 0., 1.)
            lin = tm.vec3(1., 0.6, 0.3) * dif + tm.vec3(0.91, 0.98, 1.05)
            col = tm.vec4(tm.mix(tm.vec3(1., 0.95, 0.8), tm.vec3(0.25, 0.3, 0.35), den), den)
            col.xyz *= lin
            col.xyz = tm.mix(col.xyz, bgcol, 1.0 - tm.exp(-0.003 * t * t))
            col.w *= 0.4
            col.rgb *= col.a
            sum += col * (1.0 - sum.a)
        t += tm.max(0.06, 0.05 * t)
    for i in range(40):
        pos = ro + t * rd
        if pos.y < -3. or pos.y >2. or sum.a >0.99:
            break
        den = map4(pos, i0)
        if den > 0.01:
            dif = tm.clamp((den - map4(pos + 0.3 * sundir, i0)) / 0.6, 0., 1.)
            lin = tm.vec3(1., 0.6, 0.3) * dif + tm.vec3(0.91, 0.98, 1.05)
            col = tm.vec4(tm.mix(tm.vec3(1., 0.95, 0.8), tm.vec3(0.25, 0.3, 0.35), den), den)
            col.xyz *= lin
            col.xyz = tm.mix(col.xyz, bgcol, 1.0 - tm.exp(-0.003 * t * t))
            col.w *= 0.4
            col.rgb *= col.a
            sum += col * (1.0 - sum.a)
        t += tm.max(0.06, 0.05 * t)
    for i in range(30):
        pos = ro + t * rd
        if pos.y < -3. or pos.y >2. or sum.a >0.99:
            break
        den = map3(pos, i0)
        if den > 0.01:
            dif = tm.clamp((den - map3(pos + 0.3 * sundir, i0)) / 0.6, 0., 1.)
            lin = tm.vec3(1., 0.6, 0.3) * dif + tm.vec3(0.91, 0.98, 1.05)
            col = tm.vec4(tm.mix(tm.vec3(1., 0.95, 0.8), tm.vec3(0.25, 0.3, 0.35), den), den)
            col.xyz *= lin
            col.xyz = tm.mix(col.xyz, bgcol, 1.0 - tm.exp(-0.003 * t * t))
            col.w *= 0.4
            col.rgb *= col.a
            sum += col * (1.0 - sum.a)
        t += tm.max(0.06, 0.05 * t)
    for i in range(30):
        pos = ro + t * rd
        if pos.y < -3. or pos.y >2. or sum.a >0.99:
            break
        den = map2(pos, i0)
        if den > 0.01:
            dif = tm.clamp((den - map2(pos + 0.3 * sundir, i0)) / 0.6, 0., 1.)
            lin = tm.vec3(1., 0.6, 0.3) * dif + tm.vec3(0.91, 0.98, 1.05)
            col = tm.vec4(tm.mix(tm.vec3(1., 0.95, 0.8), tm.vec3(0.25, 0.3, 0.35), den), den)
            col.xyz *= lin
            col.xyz = tm.mix(col.xyz, bgcol, 1.0 - tm.exp(-0.003 * t * t))
            col.w *= 0.4
            col.rgb *= col.a
            sum += col * (1.0 - sum.a)
        t += tm.max(0.06, 0.05 * t)

    return tm.clamp(sum, 0., 1.)   

@ti.func
def render(ro: ti.template(), rd: ti.template(), px: ti.template(), i0: ti.template(), i1: ti.template()):
    # background sky
    sun = tm.clamp(tm.dot(sundir, rd), 0., 1.)
    col = tm.vec3(0.6, 0.71, 0.75) - rd.y * 0.2 * tm.vec3(1., 0.5, 1.) + 0.15 * 0.5
    col += 0.2 * tm.vec3(1.0, 0.6, 0.1) * tm.pow(sun, 8.0)

    # clouds
    res = raymarch(ro, rd, col, px, i0, i1)
    col = col * (1.0 - res.w) + res.xyz

    # sun glare
    col += tm.vec3(0.2, 0.08, 0.04) * tm.pow(sun, 3.0)
    return col


@ti.kernel
def step(i0: ti.types.texture(num_dimensions=2), i1: ti.types.texture(num_dimensions=2)):
    for i, j in img:
        fragCoord = tm.vec2(i, j)
        p = (2. * fragCoord - iResolution) / iResolution.y
        m = iMouse[None] / iResolution

        ro = 4.0 * tm.normalize(tm.vec3(tm.sin(3.0 * m.x), 0.8 * m.y, tm.cos(3.0 * m.x))) - tm.vec3(0., 0.1, 0.)
        ta = tm.vec3(0., -1., 0.)

        ca = setCamera(ro, ta, 0.07 * tm.cos(0.25 * iTime[None]))
        rd = ca @ tm.normalize(tm.vec3(p.x, p.y, 1.5))

        img[i, j] = render(ro, rd, tm.ivec2(fragCoord - 0.5), i0, i1)



def main():
    init()
    t0 = time.perf_counter()
    gui = ti.ui.Window('Simple Cloud', res=(W, H))
    canvas = gui.get_canvas()

    tex_size = (256, 256)

    i0 = ti.Texture(ti.u8, 4, tex_size)
    i0_img = Image.open('i0.png')
    i0.from_image(i0_img)

    i1 = ti.Texture(ti.u8, 4, tex_size)
    i1_img = Image.open('i0.png')
    i1.from_image(i1_img)

    while gui.running:
        gui.get_event(ti.ui.PRESS)

        if gui.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = gui.get_cursor_pos()
            iMouse[None] = tm.vec2(mouse_x, mouse_y) * iResolution

        if gui.is_pressed(ti.ui.ESCAPE):
            gui.running = False

        step(i0, i1)
        iTime[None] = time.perf_counter() - t0
        canvas.set_image(img)
        gui.show()

if __name__ == '__main__':
    main()
