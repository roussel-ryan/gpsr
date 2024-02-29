import os, sys, argparse, math, base64, zlib, struct

if sys.version_info < (3,0):
    range = xrange

vertices = []
numVertices = [48, 48, 48, 48]
vertices_base64 = "eJyN1XtQVFUcwPEFH6DTQ9FxQZBMMIzGfDCWwsK9PkZRDJkhLZKESlE0HVGCNcC3pA4vlwVSdEUQEUQeCegoemhTSBCwRAVNRsRVUGJhFtlKyCY5v7Nzf9up9h9mPsOce+7v912wrnJb4PywlRw6+PcnV5C9/HQJ1v/obfSnTIad/r7w/7xLKFZtyt8/8pqZF0n8IPPgbbWqpcoa5oplcvsPNN1CRsPxysEFV5kPW+2fM3KWQZg6z6jfXlbJ/NrLTw/yXKGWes9+R/216Q/M3O3mhrVVqmbkvwnlYd0LLNdpCfYIp6lND7e2mbnDoIVHo1J0BJ/fucJjVUXez2beIfGDQg31SkEX88W468wDa41xyfMMwpqTG97aPa6W+UUnZWF8TjfMk+B5Fkucu3fy727WA3O09/9wsx6Yox6Yox4IpwfC6YFweiCcHginB8LpgXB6IJweCKcHwumBcHognB4Ipwc8f+7eZZIPcwE7zA0E7g/vi/3NZx36Bud8IYk+t2F06JkxxUPEoktpTncKtMgHiWcL0qPvbmxm9wSfXl9huTChmxxHHhRUcrTf0kCy0fm3BnvH1W1vJxqOH6PzAe9cY9hR6PWEeWnVxPk9Xw8Vjyf3l8wtNrnD9blxy7ytxIjGkOdJt5+QDOruUfV2MXJr5kfoc8FD+y/55N41sPtjz5T4cLHabZHfuzdaSRryV72S23ufBQvxHE9E5yscly9KO9JEUjl+gN7fg/qKUZ/GH5I3ERX0FjpqTtlcK7HapXDC73tvM49T+/u7fTVUPGEd4HH4eQNzmCc4PBf8DdoDnYOAe5C6WQ8C7iEJOfSQiM6HvadyPHng/gLuARz3AI57UFPHPdA9CnjvSRxXSdzUwxHkBs+BvedwPBudD3vXcDyL3h/3kEkd9wCOewDHPcBzwcfTHjrbF8S/p1ErwGGP1MvBYT6XNW3VMypU5XB/Ge3fdkbOkvejUxXga6XfO+HyyrqopLyRzDMlPkKsoXPGbkHneYzjWeh8L8k8zT2fzgf8Uc7k29OeNJEC6v7uaq2TjY04v3+25yf1d0gh9VivLF2D0kbceuXbSQ4+v5Ai6sY0i8If7pkc3hd8Y/x5mzFD88y80tnnS1cXLTkh8VGie9L8B53r20gO8ox3LqqHvd1r5uHpV40flvTA32F2/pzWxzsTJv1q5rOpF6P7F9xQvH6/5Sl7L/uPXwkqjbIRdfd0AYljn7I5nFmf/3jzaBsxkn6/TqN5gvP2Dv93sKskbuoB+wvFwN6TOX4Ane8l+Xtr7nsH7k9wD7HUcQ97qOMedlPHPcD74h6wQw8JEjf1EI8cesAOPSSi86EH7NDDLnR/6AHeC/cAc8A9fIPmCc7be49W7rjMRa0Al9HvdePlRzOadhxgDnNzXZ59/o5nigLuCe/rHXAr86PWNObKlT4WLSMekGj63Mn6nTftd9mKir0/Juh9m0mMxOVi8pDFVnV7uun9TS5v17nej3zO+gGft6WpSt3Rx+YJ5/s5LNzwZ24v2cdxJZ0PeGCdY3untZFEUtd4t2iGt9mKZ1uzLGfNMZJw6j8t7vr+gq+dOD62MTtoqZFsoh48ocx6SZnJY+lzwX3lDzeLV/pJHMf3SXws/X9nINuQx2UEB8zUtpIIjm9B54e3XPijz68LnWPyVej+5Suia2KmdJEQ6mF2jYfXLbYTC6uOteSl6Uko9WFTlBNddbbiuHO96/v8OpnDPMF3ovlDD+cGXMA9nJW4qYfTyKGHLOTQQzY6H/Z+iuPlA/cXcA8XqeMeKqjjHrTUcQ8F9Ll47yc5fkriph5KkcPeL3H8Ajof9l7K8Xp0f+ihjjruoYY67gEc9/Admj/00Bf29ITLi9MKcNjjzNeU9kGrU5jDfIr0KZ9/ZpGugHvC+wZO81w1OqSI+V9ipmIS"
triangles = [[ 2, 1, 0, 2, 4, 3, 2, 5, 4, 2, 0, 21, 2, 21, 5, 21, 20, 5, 5, 20, 6, 13, 12, 11, 13, 15, 14, 13, 16, 15, 13, 11, 10, 13, 10, 16, 10, 9, 16, 16, 9, 17, 24, 22, 23, 24, 25, 26, 24, 26, 27, 24, 43, 22, 24, 27, 43, 43, 27, 42, 27, 28, 42, 35, 33, 34, 35, 36, 37, 35, 37, 38, 35, 32, 33, 35, 38, 32, 32, 38, 31, 38, 39, 31, 6, 20, 7, 20, 19, 7, 28, 29, 42, 42, 29, 41, 7, 19, 8, 19, 18, 8, 29, 30, 41, 41, 30, 40, 8, 18, 9, 18, 17, 9, 30, 31, 40, 40, 31, 39, 0, 1, 23, 0, 23, 22, 1, 2, 24, 1, 24, 23, 2, 3, 25, 2, 25, 24, 3, 4, 26, 3, 26, 25, 6, 7, 29, 6, 29, 28, 7, 8, 30, 7, 30, 29, 8, 9, 31, 8, 31, 30, 11, 12, 34, 11, 34, 33, 12, 13, 35, 12, 35, 34, 13, 14, 36, 13, 36, 35, 14, 15, 37, 14, 37, 36, 17, 18, 40, 17, 40, 39, 18, 19, 41, 18, 41, 40, 19, 20, 42, 19, 42, 41, 21, 0, 20, 44, 20, 0, 44, 42, 20, 44, 22, 42, 43, 42, 22, 4, 5, 6, 4, 6, 45, 45, 6, 28, 26, 45, 28, 26, 28, 27, 10, 11, 9, 9, 11, 46, 9, 46, 31, 31, 46, 33, 32, 31, 33, 17, 15, 16, 17, 47, 15, 47, 17, 39, 37, 47, 39, 37, 39, 38], [ 2, 1, 0, 2, 4, 3, 2, 5, 4, 2, 0, 21, 2, 21, 5, 21, 20, 5, 5, 20, 6, 13, 12, 11, 13, 15, 14, 13, 16, 15, 13, 11, 10, 13, 10, 16, 10, 9, 16, 16, 9, 17, 24, 22, 23, 24, 25, 26, 24, 26, 27, 24, 43, 22, 24, 27, 43, 43, 27, 42, 27, 28, 42, 35, 33, 34, 35, 36, 37, 35, 37, 38, 35, 32, 33, 35, 38, 32, 32, 38, 31, 38, 39, 31, 6, 20, 7, 20, 19, 7, 28, 29, 42, 42, 29, 41, 7, 19, 8, 19, 18, 8, 29, 30, 41, 41, 30, 40, 8, 18, 9, 18, 17, 9, 30, 31, 40, 40, 31, 39, 0, 1, 23, 0, 23, 22, 1, 2, 24, 1, 24, 23, 2, 3, 25, 2, 25, 24, 3, 4, 26, 3, 26, 25, 6, 7, 29, 6, 29, 28, 7, 8, 30, 7, 30, 29, 8, 9, 31, 8, 31, 30, 11, 12, 34, 11, 34, 33, 12, 13, 35, 12, 35, 34, 13, 14, 36, 13, 36, 35, 14, 15, 37, 14, 37, 36, 17, 18, 40, 17, 40, 39, 18, 19, 41, 18, 41, 40, 19, 20, 42, 19, 42, 41, 21, 0, 20, 44, 20, 0, 44, 42, 20, 44, 22, 42, 43, 42, 22, 4, 5, 6, 4, 6, 45, 45, 6, 28, 26, 45, 28, 26, 28, 27, 10, 11, 9, 9, 11, 46, 9, 46, 31, 31, 46, 33, 32, 31, 33, 17, 15, 16, 17, 47, 15, 47, 17, 39, 37, 47, 39, 37, 39, 38], [ 2, 1, 0, 2, 4, 3, 2, 5, 4, 2, 0, 21, 2, 21, 5, 21, 20, 5, 5, 20, 6, 13, 12, 11, 13, 15, 14, 13, 16, 15, 13, 11, 10, 13, 10, 16, 10, 9, 16, 16, 9, 17, 24, 22, 23, 24, 25, 26, 24, 26, 27, 24, 43, 22, 24, 27, 43, 43, 27, 42, 27, 28, 42, 35, 33, 34, 35, 36, 37, 35, 37, 38, 35, 32, 33, 35, 38, 32, 32, 38, 31, 38, 39, 31, 6, 20, 7, 20, 19, 7, 28, 29, 42, 42, 29, 41, 7, 19, 8, 19, 18, 8, 29, 30, 41, 41, 30, 40, 8, 18, 9, 18, 17, 9, 30, 31, 40, 40, 31, 39, 0, 1, 23, 0, 23, 22, 1, 2, 24, 1, 24, 23, 2, 3, 25, 2, 25, 24, 3, 4, 26, 3, 26, 25, 6, 7, 29, 6, 29, 28, 7, 8, 30, 7, 30, 29, 8, 9, 31, 8, 31, 30, 11, 12, 34, 11, 34, 33, 12, 13, 35, 12, 35, 34, 13, 14, 36, 13, 36, 35, 14, 15, 37, 14, 37, 36, 17, 18, 40, 17, 40, 39, 18, 19, 41, 18, 41, 40, 19, 20, 42, 19, 42, 41, 21, 0, 20, 44, 20, 0, 44, 42, 20, 44, 22, 42, 43, 42, 22, 4, 5, 6, 4, 6, 45, 45, 6, 28, 26, 45, 28, 26, 28, 27, 10, 11, 9, 9, 11, 46, 9, 46, 31, 31, 46, 33, 32, 31, 33, 17, 15, 16, 17, 47, 15, 47, 17, 39, 37, 47, 39, 37, 39, 38], [ 2, 1, 0, 2, 4, 3, 2, 5, 4, 2, 0, 21, 2, 21, 5, 21, 20, 5, 5, 20, 6, 13, 12, 11, 13, 15, 14, 13, 16, 15, 13, 11, 10, 13, 10, 16, 10, 9, 16, 16, 9, 17, 24, 22, 23, 24, 25, 26, 24, 26, 27, 24, 43, 22, 24, 27, 43, 43, 27, 42, 27, 28, 42, 35, 33, 34, 35, 36, 37, 35, 37, 38, 35, 32, 33, 35, 38, 32, 32, 38, 31, 38, 39, 31, 6, 20, 7, 20, 19, 7, 28, 29, 42, 42, 29, 41, 7, 19, 8, 19, 18, 8, 29, 30, 41, 41, 30, 40, 8, 18, 9, 18, 17, 9, 30, 31, 40, 40, 31, 39, 0, 1, 23, 0, 23, 22, 1, 2, 24, 1, 24, 23, 2, 3, 25, 2, 25, 24, 3, 4, 26, 3, 26, 25, 6, 7, 29, 6, 29, 28, 7, 8, 30, 7, 30, 29, 8, 9, 31, 8, 31, 30, 11, 12, 34, 11, 34, 33, 12, 13, 35, 12, 35, 34, 13, 14, 36, 13, 36, 35, 14, 15, 37, 14, 37, 36, 17, 18, 40, 17, 40, 39, 18, 19, 41, 18, 41, 40, 19, 20, 42, 19, 42, 41, 21, 0, 20, 44, 20, 0, 44, 42, 20, 44, 22, 42, 43, 42, 22, 4, 5, 6, 4, 6, 45, 45, 6, 28, 26, 45, 28, 26, 28, 27, 10, 11, 9, 9, 11, 46, 9, 46, 31, 31, 46, 33, 32, 31, 33, 17, 15, 16, 17, 47, 15, 47, 17, 39, 37, 47, 39, 37, 39, 38]]
decoration = [[ 0.223551, 0.000000, 1.198843, -0.276449, 0.000000, 1.198843, 0.000000, 0.000000, 0.750000, 0.000000, 0.000000, 1.000000, -0.052898, 0.000000, 1.300000, -0.143891, 0.000000, 1.550000], [ -0.843080, -0.000000, 2.910312, -0.343080, 0.000000, 2.910312, -0.475638, -0.000000, 2.461469, -0.566631, 0.000000, 2.711469, -0.619529, 0.000000, 3.011469, -0.619529, 0.000000, 3.261469], [ -0.395978, 0.000000, 4.615410, -0.895978, 0.000000, 4.615410, -0.619529, 0.000000, 4.166567, -0.619529, 0.000000, 4.416567, -0.672427, 0.000000, 4.716567, -0.763420, 0.000000, 4.966567], [ -1.462609, 0.000000, 6.326879, -0.962609, 0.000000, 6.326879, -1.095167, 0.000000, 5.878036, -1.186160, 0.000000, 6.128036, -1.239058, 0.000000, 6.428036, -1.239058, 0.000000, 6.678036]]

color = [1, 1, 1, 1]

index_base64 = 'eJylVlFv2zYQfs+vYGUMUACLcjIUKBLbgOMGWIBkCdJgQDfsgSZPFguJ9CjKrjf0v/dIKrYk28G2CrAl6u6+++6Od9T43cfH+cvnp1uS27KYno3DjeA1zoGJ8OiXJViGWnaVwF+1XE+iuVYWlE1etiuICA+rSWThq00dzDXhOTMV2Elts+RDRNLp2R7OSlvA9IYttoVW9EtFKlauCkAcAeM0SPfa75KEtHSTpCWruJErSyrDJ5Hjd5WmXCi6COpfKsp1meZMCVpKJTMJIrmglwgTTcdpMP6vaJwp5Zn8fwgtS/1DAItdOo5j2G07g+5yRRmShRZb8k9H4C69BpMVenNFcikEqOsDjY0UNr8iF6PRT4fCHOQyt6ekJTNLqa7I6FC0YkJItTyQfTvrLAcGlAAzZ2rNqiP0f4Cc1TXPE8at1EgR6wp9Jq9PmOZ9VsdpaJCxS2jzigd6UkyiNl9XoCBq9PrVWjNDGtsJEZrXJbYSXYK9LcA93mzvRNyFPL/uWIPCBANaK9iQm9nN5/vHX+mtfxkH5CGxpgY063o1wCx84uCNs1r5NJD4vJdjp1s1Wm0X3jIO3tvY7krTma8tYaRwBTjIvAP1kh7ok5bK3jtBHD2WSkbDjvg34Fabn+PRkCRY0iEZnQ8DufPD6qap90CFzLK66tOf68Ij0Q+I0vwdAQkQ1Qp4XSDnExgXzjz8vUFkaXSthDc6AYQILqaT2VRkZjh51hYrR+asBMOOppZ7Uc8JmgbLYBhH4R452uj2gdmcPt0NyXtc9BJOfwej4xO5Ds4os5bx3B0NRhe7nZexojpWnMZokwMUTwa4rNzmm5DLUS94F065LaHKUToYfJy9zAaD615+yEsOJJOmsjhVDEJb8G1FFkCw9AI7nbgaymxLNrnkOfGA+FaWK20sJb+AAbKBZk1YUXgVqDqeOnv/XjNsSnrnLR5QOY4wl+4X+DbZGrZ6C8vx4FH7TdaE8QkssRiKxamJjzrzq6aayNbuwgw8MTLn6VR2Gxi/DYLfP0Z/9vJ7kGvcIEay4qLf7hbPUWbEQyOP/XFfG7iITrfgPrQdLN1IA5mrETpwc+nQqs2Wvhqi9g7jX5pQKGVVyTW81XL00u3/8NeL4Fu/EQ1gwCpEu1fF86ozWF+HZWvAxm2kMDKpqdWzH+v3Wq/i0/PXw9FwAsQthh12uHeeoZJ/w+7NRiqhNxTP2Ns1uJla4UcaIuBZ4vSi4Rsj/5Wh1+z7dPf9V8c4DYfg2H/4Tb8DrvXVsg=='

def decodeVertices():
    vertices_binary = zlib.decompress(base64.b64decode(vertices_base64))
    k = 0
    for i in range(len(numVertices)):
        current = []
        for j in range(numVertices[i] * 3):
            current.append(float(struct.unpack('=d', vertices_binary[k:k+8])[0]))
            k += 8
        vertices.append(current)

def dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def normalize(a):
    length = math.sqrt(dot(a, a))
    for i in range(3):
        a[i]/=length
    return a

def distance(a, b):
    c = [0] * 2
    for i in range(len(a)):
        c[i] = a[i] - b[i]
    return math.sqrt(dot(c, c))

def cross(a, b):
    c = [0, 0, 0]
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

class Quaternion:
    def __init__(self, values):
        self.R = values

    def __str__(self):
        return str(self.R)

    def __mul__(self, other):
        imagThis = self.imag()
        imagOther = other.imag()
        cro = cross(imagThis, imagOther)

        ret = ([self.R[0] * other.R[0] - dot(imagThis, imagOther)] +
               [self.R[0] * imagOther[0] + other.R[0] * imagThis[0] + cro[0],
                self.R[0] * imagOther[1] + other.R[0] * imagThis[1] + cro[1],
                self.R[0] * imagOther[2] + other.R[0] * imagThis[2] + cro[2]])

        return Quaternion(ret)

    def imag(self):
        return self.R[1:]

    def conjugate(self):
        ret = [0] * 4
        ret[0] = self.R[0]
        ret[1] = -self.R[1]
        ret[2] = -self.R[2]
        ret[3] = -self.R[3]

        return Quaternion(ret)

    def inverse(self):
        return self.conjugate()

    def rotate(self, vec3D):
        quat = Quaternion([0.0] + vec3D)
        conj = self.conjugate()

        ret = self * (quat * conj)
        return ret.R[1:]

def getQuaternion(u, ref):
    u = normalize(u)
    ref = normalize(ref)
    axis = cross(u, ref)
    normAxis = math.sqrt(dot(axis, axis))

    if normAxis < 1e-12:
        if math.fabs(dot(u, ref) - 1.0) < 1e-12:
            return Quaternion([1.0, 0.0, 0.0, 0.0])

        return Quaternion([0.0, 1.0, 0.0, 0.0])

    axis = normalize(axis)
    cosAngle = math.sqrt(0.5 * (1 + dot(u, ref)))
    sinAngle = math.sqrt(1 - cosAngle * cosAngle)

    return Quaternion([cosAngle, sinAngle * axis[0], sinAngle * axis[1], sinAngle * axis[2]])

def exportVTK():
    vertices_str = ""
    triangles_str = ""
    color_str = ""
    cellTypes_str = ""
    startIdx = 0
    vertexCounter = 0
    cellCounter = 0
    lookup_table = []
    lookup_table.append([0.5, 0.5, 0.5, 0.5])
    lookup_table.append([1.0, 0.847, 0.0, 1.0])
    lookup_table.append([1.0, 0.0, 0.0, 1.0])
    lookup_table.append([0.537, 0.745, 0.525, 1.0])
    lookup_table.append([0.5, 0.5, 0.0, 1.0])
    lookup_table.append([1.0, 0.541, 0.0, 1.0])
    lookup_table.append([0.0, 0.0, 1.0, 1.0])

    decodeVertices()

    for i in range(len(vertices)):
        for j in range(0, len(vertices[i]), 3):
            vertices_str += ("%f %f %f\n" %(vertices[i][j], vertices[i][j+1], vertices[i][j+2]))
            vertexCounter += 1

        for j in range(0, len(triangles[i]), 3):
            triangles_str += ("3 %d %d %d\n" % (triangles[i][j] + startIdx, triangles[i][j+1] + startIdx, triangles[i][j+2] + startIdx))
            cellTypes_str += "5\n"
            tmp_color = lookup_table[color[i]]
            color_str += ("%f %f %f %f\n" % (tmp_color[0], tmp_color[1], tmp_color[2], tmp_color[3]))
            cellCounter += 1
        startIdx = vertexCounter

    fh = open('lattice_1e5par_1nc_csroff_ElementPositions.vtk','w')
    fh.write("# vtk DataFile Version 2.0\n")
    fh.write("test\nASCII\n\n")
    fh.write("DATASET UNSTRUCTURED_GRID\n")
    fh.write("POINTS " + str(vertexCounter) + " float\n")
    fh.write(vertices_str)
    fh.write("CELLS " + str(cellCounter) + " " + str(cellCounter * 4) + "\n")
    fh.write(triangles_str)
    fh.write("CELL_TYPES " + str(cellCounter) + "\n")
    fh.write(cellTypes_str)
    fh.write("CELL_DATA " + str(cellCounter) + "\n")
    fh.write("COLOR_SCALARS type 4\n")
    fh.write(color_str + "\n")
    fh.close()

def getNormal(tri_vertices):
    vec1 = [tri_vertices[1][0] - tri_vertices[0][0],
            tri_vertices[1][1] - tri_vertices[0][1],
            tri_vertices[1][2] - tri_vertices[0][2]]
    vec2 = [tri_vertices[2][0] - tri_vertices[0][0],
            tri_vertices[2][1] - tri_vertices[0][1],
            tri_vertices[2][2] - tri_vertices[0][2]]
    return normalize(cross(vec1,vec2))

def exportWeb(bgcolor):
    lookup_table = []
    lookup_table.append([0.5, 0.5, 0.5])
    lookup_table.append([1.0, 0.847, 0.0])
    lookup_table.append([1.0, 0.0, 0.0])
    lookup_table.append([0.537, 0.745, 0.525])
    lookup_table.append([0.5, 0.5, 0.0])
    lookup_table.append([1.0, 0.541, 0.0])
    lookup_table.append([0.0, 0.0, 1.0])

    decodeVertices()

    mesh = "'data:"
    mesh += "{"
    mesh += "\"autoClear\":true,"
    mesh += "\"clearColor\":[0.0000,0.0000,0.0000],"
    mesh += "\"ambientColor\":[0.0000,0.0000,0.0000],"
    mesh += "\"gravity\":[0.0000,-9.8100,0.0000],"
    mesh += "\"cameras\":["
    mesh += "{"
    mesh += "\"name\":\"Camera\","
    mesh += "\"id\":\"Camera\","
    mesh += "\"position\":[21.7936,2.2312,-85.7292],"
    mesh += "\"rotation\":[0.0432,-0.1766,-0.0668],"
    mesh += "\"fov\":0.8578,"
    mesh += "\"minZ\":10.0000,"
    mesh += "\"maxZ\":10000.0000,"
    mesh += "\"speed\":1.0000,"
    mesh += "\"inertia\":0.9000,"
    mesh += "\"checkCollisions\":false,"
    mesh += "\"applyGravity\":false,"
    mesh += "\"ellipsoid\":[0.2000,0.9000,0.2000]"
    mesh += "}],"
    mesh += "\"activeCamera\":\"Camera\","
    mesh += "\"lights\":["
    mesh += "{"
    mesh += "\"name\":\"Lamp\","
    mesh += "\"id\":\"Lamp\","
    mesh += "\"type\":0.0000,"
    mesh += "\"position\":[4.0762,34.9321,-63.5788],"
    mesh += "\"intensity\":1.0000,"
    mesh += "\"diffuse\":[1.0000,1.0000,1.0000],"
    mesh += "\"specular\":[1.0000,1.0000,1.0000]"
    mesh += "}],"
    mesh += "\"materials\":[],"
    mesh += "\"meshes\":["

    for i in range(len(triangles)):
        vertex_list = []
        indices_list = []
        normals_list = []
        color_list = []
        for j in range(0, len(triangles[i]), 3):
            tri_vertices = []
            idcs = triangles[i][j:j + 3]
            tri_vertices.append(vertices[i][3 * idcs[0]:3 * (idcs[0] + 1)])
            tri_vertices.append(vertices[i][3 * idcs[1]:3 * (idcs[1] + 1)])
            tri_vertices.append(vertices[i][3 * idcs[2]:3 * (idcs[2] + 1)])
            indices_list.append(','.join(str(n) for n in range(len(vertex_list),len(vertex_list) + 3)))
            # left hand order!
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[0]))
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[2]))
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[1]))
            normal = getNormal(tri_vertices)
            normals_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in normal * 3))
            color_list.append(','.join([str(n) for n in lookup_table[color[i]]] * 3))
        mesh += "{"
        mesh += "\"name\":\"element_" + str(i) + "\","
        mesh += "\"id\":\"element_" + str(i) + "\","
        mesh += "\"position\":[0.0,0.0,0.0],"
        mesh += "\"rotation\":[0.0,0.0,0.0],"
        mesh += "\"scaling\":[1.0,1.0,1.0],"
        mesh += "\"isVisible\":true,"
        mesh += "\"isEnabled\":true,"
        mesh += "\"useFlatShading\":false,"
        mesh += "\"checkCollisions\":false,"
        mesh += "\"billboardMode\":0,"
        mesh += "\"receiveShadows\":false,"
        mesh += "\"positions\":[" + ','.join(vertex_list) + "],"
        mesh += "\"normals\":[" + ','.join(normals_list) + "],"
        mesh += "\"indices\":[" + ','.join(indices_list) + "],"
        mesh += "\"colors\":[" + ','.join(color_list) + "],"
        mesh += "\"subMeshes\":["
        mesh += "{"
        mesh += "\"materialIndex\":0,"
        mesh += " \"verticesStart\":0,"
        mesh += " \"verticesCount\":" + str(len(triangles[i])) + ","
        mesh += " \"indexStart\":0,"
        mesh += " \"indexCount\":" + str(len(triangles[i])) + ""
        mesh += "}]"
        mesh += "}"
        mesh += ","

        del normals_list[:]
        del vertex_list[:]
        del color_list[:]
        del indices_list[:]

    mesh = mesh[:-1] + "]"
    mesh += "}'"
    index_compressed = base64.b64decode(index_base64)
    index = str(zlib.decompress(index_compressed))
    if (len(bgcolor) == 3):
        mesh += ";\n            "
        mesh += "scene.clearColor = new BABYLON.Color3(%f, %f, %f)" % (bgcolor[0], bgcolor[1], bgcolor[2])

    index = index.replace('##DATA##', mesh)
    fh = open('lattice_1e5par_1nc_csroff_ElementPositions.html','w')
    fh.write(index)
    fh.close()

def computeMinAngle(idx, curAngle, positions, connections, check):
    matrix = [[-math.cos(curAngle), -math.sin(curAngle)], [math.sin(curAngle), -math.cos(curAngle)]]

    minAngle = 2 * math.pi
    nextIdx = -1

    for j in connections[idx]:
        direction = [positions[j][0] - positions[idx][0],
                     positions[j][1] - positions[idx][1]]
        direction = [dot(matrix[0],direction), dot(matrix[1],direction)]

        if math.fabs(dot([1.0, 0.0], direction) / distance(positions[j], positions[idx]) - 1.0) < 1e-6: continue

        angle = math.fmod(math.atan2(direction[1],direction[0]) + 2 * math.pi, 2 * math.pi)

        if angle < minAngle:
            nextIdx = j
            minAngle = angle
        if angle == minAngle and check:
            dire =  [positions[j][0] - positions[idx][0],
                     positions[j][1] - positions[idx][1]]
            minA0 = math.atan2(dire[1], dire[0])
            minA1 = computeMinAngle(nextIdx, minA0, positions, connections, False)
            minA2 = computeMinAngle(j, minA0, positions, connections, False)
            if minA2[1] < minA2[1]:
                nextIdx = j

    if nextIdx == -1:
        nextIdx = connections[idx][0]

    return (nextIdx, minAngle)

def squashVertices(positionDict, connections):
    removedItems = []
    indices = [int(k) for k in positionDict.keys()]
    idxChanges = indices
    for i in indices:
        if i in removedItems:
            continue
        for j in indices:
            if j in removedItems or j == i:
                continue

            if distance(positionDict[i], positionDict[j]) < 1e-6:
                connections[j] = list(set(connections[j]))
                if i in connections[j]:
                    connections[j].remove(i)
                if j in connections[i]:
                    connections[i].remove(j)

                connections[i].extend(connections[j])
                connections[i] = list(set(connections[i]))
                connections[i].sort()

                for k in connections.keys():
                    if k == i: continue
                    if j in connections[k]:
                        idx = connections[k].index(j)
                        connections[k][idx] = i

                idxChanges[j] = i
                removedItems.append(j)

    for i in removedItems:
        del positionDict[i]
        del connections[i]

    for i in connections.keys():
        connections[i] = list(set(connections[i]))
        connections[i].sort()

    return idxChanges

def computeLineEquations(positions, connections):
    cons = set()
    for i in connections.keys():
        for j in connections[i]:
            cons.add((min(i, j), max(i, j)))

    lineEquations = {}
    for item in cons:
        a = (positions[item[1]][1] - positions[item[0]][1])
        b = -(positions[item[1]][0] - positions[item[0]][0])

        xm = 0.5 * (positions[item[0]][0] + positions[item[1]][0])
        ym = 0.5 * (positions[item[0]][1] + positions[item[1]][1])
        c = -(a * xm +  b * ym)

        lineEquations[item] = (a, b, c)

    return lineEquations

def checkPossibleSegmentIntersection(segment, positions, lineEquations, minAngle, lastIntersection):
    item1 = (min(segment), max(segment))

    (a1, b1, c1) = (0,0,0)
    A = [0]*2
    B = A

    if segment[0] == None:
        A = lastIntersection
        B = positions[segment[1]]

        a1 = B[1] - A[1]
        b1 = -(B[0] - A[0])
        xm = 0.5 * (A[0] + B[0])
        ym = 0.5 * (A[1] + B[1])
        c1 = -(a1 * xm + b1 * ym)

    else:
        A = positions[segment[0]]
        B = positions[segment[1]]

        (a1, b1, c1) = lineEquations[item1]

        if segment[1] < segment[0]:
            (a1, b1, c1) = (-a1, -b1, -c1)

    curAngle = math.atan2(a1, -b1)
    matrix = [[-math.cos(curAngle), -math.sin(curAngle)], [math.sin(curAngle), -math.cos(curAngle)]]

    origMinAngle = minAngle

    segment1 = [B[0] - A[0], B[1] - A[1], 0.0]

    intersectingSegments = []
    distanceAB = distance(A, B)
    for item2 in lineEquations.keys():
        item = item2
        C = positions[item[0]]
        D = positions[item[1]]

        if segment[0] == None:
            if (segment[1] == item[0] or
                segment[1] == item[1]): continue
        else:
            if (item1[0] == item[0] or
                item1[1] == item[0] or
                item1[0] == item[1] or
                item1[1] == item[1]): continue

        nodes = set([item1[0],item1[1],item[0],item[1]])
        if len(nodes) < 4: continue       # share same vertex

        (a2, b2, c2) = lineEquations[item]

        segment2 = [C[0] - A[0], C[1] - A[1], 0.0]
        segment3 = [D[0] - A[0], D[1] - A[1], 0.0]

        # check that C and D aren't on the same side of AB
        if cross(segment1, segment2)[2] * cross(segment1, segment3)[2] > 0.0: continue

        if cross(segment1, segment2)[2] < 0.0 or cross(segment1, segment3)[2] > 0.0:
            (C, D, a2, b2, c2) = (D, C, -a2, -b2, -c2)
            item = (item[1], item[0])

        denominator = a1 * b2 - b1 * a2
        if math.fabs(denominator) < 1e-9: continue

        px = (b1 * c2 - b2 * c1) / denominator
        py = (a2 * c1 - a1 * c2) / denominator

        distanceCD = distance(C, D)

        distanceAP = distance(A, [px, py])
        distanceBP = distance(B, [px, py])
        distanceCP = distance(C, [px, py])
        distanceDP = distance(D, [px, py])

        # check intersection is between AB and CD
        check1 = (distanceAP - 1e-6 < distanceAB and distanceBP - 1e-6 < distanceAB)
        check2 = (distanceCP - 1e-6 < distanceCD and distanceDP - 1e-6 < distanceCD)
        if not check1 or not check2: continue

        if math.fabs(dot(segment1, [D[0] - C[0], D[1] - C[1], 0.0]) / (distanceAB * distanceCD) + 1.0) < 1e-9: continue

        if ((distanceAP < 1e-6) or
            (distanceBP < 1e-6 and distanceCP < 1e-6) or
            (distanceDP < 1e-6)):
            continue

        direction = [D[0] - C[0], D[1] - C[1]]
        direction = [dot(matrix[0], direction), dot(matrix[1], direction)]
        angle = math.fmod(math.atan2(direction[1], direction[0]) + 2 * math.pi, 2 * math.pi)

        newSegment = ([px, py], item[1], distanceAP, angle)

        if distanceCP < 1e-6 and angle > origMinAngle: continue
        if distanceBP > 1e-6 and angle > math.pi: continue

        if len(intersectingSegments) == 0:
            intersectingSegments.append(newSegment)
            minAngle = angle
        else:
            if intersectingSegments[0][2] - 1e-9 > distanceAP:
                intersectingSegments[0] = newSegment
                minAngle = angle
            elif intersectingSegments[0][2] + 1e-9 > distanceAP and angle < minAngle:
                intersectingSegments[0] = newSegment
                minAngle = angle

    return intersectingSegments

def projectToPlane(normal):
    fh = open('lattice_1e5par_1nc_csroff_ElementPositions.gpl','w')
    normal = normalize(normal)
    ori = getQuaternion(normal, [0, 0, 1])

    left2Right = [0, 0, 1]
    if math.fabs(math.fabs(dot(normal, left2Right)) - 1) < 1e-9:
        left2Right = [1, 0, 0]
    rotL2R = ori.rotate(left2Right)
    deviation = math.atan2(rotL2R[1], rotL2R[0])
    rotBack = Quaternion([math.cos(0.5 * deviation), 0, 0, -math.sin(0.5 * deviation)])
    ori = rotBack * ori

    decodeVertices()

    for i in range(len(vertices)):
        positions = {}
        connections = {}
        for j in range(0, len(vertices[i]), 3):
            nextPos3D = ori.rotate(vertices[i][j:j+3])
            nextPos2D = nextPos3D[0:2]
            positions[j/3] = nextPos2D

        if len(positions) == 0:
            continue

        idx = 0
        maxX = positions[0][0]
        for j in positions.keys():
            if positions[j][0] > maxX:
                maxX = positions[j][0]
                idx = j
            if positions[j][0] == maxX and positions[j][1] > positions[idx][1]:
                idx = j

        for j in range(0, len(triangles[i]), 3):
            for k in range(0, 3):
                vertIdx = triangles[i][j + k]
                if not vertIdx in connections:
                    connections[vertIdx] = []
                for l in range(1, 3):
                    connections[vertIdx].append(triangles[i][j + ((k + l) % 3)])

        numConnections = 0
        for j in connections.keys():
            connections[j] = list(set(connections[j]))
            numConnections += len(connections[j])

        numConnections /= 2
        idChanges = squashVertices(positions, connections)

        lineEquations = computeLineEquations(positions, connections)

        idx = idChanges[int(idx)]
        fh.write("%.6f    %.6f\n" % (positions[idx][0], positions[idx][1]))

        curAngle = math.pi
        origin = idx
        count = 0
        passed = []
        while (count == 0 or distance(positions[origin], positions[idx]) > 1e-9) and not count > numConnections:
            nextGen = computeMinAngle(idx, curAngle, positions, connections, False)
            nextIdx = nextGen[0]
            direction = [positions[nextIdx][0] - positions[idx][0],
                         positions[nextIdx][1] - positions[idx][1]]
            curAngle = math.atan2(direction[1], direction[0])

            intersections = checkPossibleSegmentIntersection((idx, nextIdx), positions, lineEquations, nextGen[1], [])
            if len(intersections) > 0:
                while len(intersections) > 0 and not count > numConnections:
                    fh.write("%.6f    %.6f\n" %(intersections[0][0][0], intersections[0][0][1]))
                    count += 1
    
                    idx = intersections[0][1]
                    direction = [positions[idx][0] - intersections[0][0][0],
                                 positions[idx][1] - intersections[0][0][1]]
                    curAngle = math.atan2(direction[1], direction[0])
                    nextGen = computeMinAngle(idx, curAngle, positions, connections, False)
    
                    intersections = checkPossibleSegmentIntersection((None, idx), positions, lineEquations, nextGen[1], intersections[0][0])
            else:
                idx = nextIdx
    
            fh.write("%.6f    %.6f\n" % (positions[idx][0], positions[idx][1]))
    
            if idx in passed:
                direction1 = [positions[idx][0] - positions[passed[-1]][0],
                              positions[idx][1] - positions[passed[-1]][1]]
                direction2 = [positions[origin][0] - positions[passed[-1]][0],
                              positions[origin][1] - positions[passed[-1]][1]]
                dist1 = distance(positions[idx], positions[passed[-1]])
                dist2 = distance(positions[origin], positions[passed[-1]])
                if dist1 * dist2 > 0.0 and math.fabs(math.fabs(dot(direction1, direction2) / (dist1 * dist2)) - 1.0) > 1e-9:
                    sys.stderr.write("error: projection cycling on element id: %d, vertex id: %d\n" %(i, idx))
                break

            passed.append(idx)
            count += 1
        fh.write("\n")

        if count > numConnections:
            sys.stderr.write("error: projection cycling on element id: %d\n" % i)

        for j in range(0, len(decoration[i]), 6):
            for k in range(j, j + 6, 3):
                nextPos3D = ori.rotate(decoration[i][k:k+3])
                fh.write("%.6f    %.6f\n" % (nextPos3D[0], nextPos3D[1]))
            fh.write("\n")

    fh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-vtk', action='store_true')
    parser.add_argument('--export-web', action='store_true')
    parser.add_argument('--background', nargs=3, type=float)
    parser.add_argument('--project-to-plane', action='store_true')
    parser.add_argument('--normal', nargs=3, type=float)
    args = parser.parse_args()

    if (args.export_vtk):
        exportVTK()
        sys.exit()

    if (args.export_web):
        bgcolor = []
        if (args.background):
            validBackground = True
            for comp in bgcolor:
                if comp < 0.0 or comp > 1.0:
                    validBackground = False
                    break
            if (validBackground):
                bgcolor = args.background
        exportWeb(bgcolor)
        sys.exit()

    if (args.project_to_plane):
        normal = [0.0, 1.0, 0.0]
        if (args.normal):
            normal = args.normal
        projectToPlane(normal)
        sys.exit()

    parser.print_help()
