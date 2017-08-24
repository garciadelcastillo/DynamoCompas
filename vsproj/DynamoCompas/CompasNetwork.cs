using System.Collections.Generic;
using Autodesk.DesignScript.Interfaces;
using Dynamo.Graph.Nodes;
using Autodesk.DesignScript.Runtime;
using System;

/*
 Autodesk ETH Workshop Zurich
 August 2017
 */

namespace Compas.Dynamo.Datastructures
{
    // Add the IsDesignScriptCompatible attribute to ensure
    // that it gets loaded in Dynamo.
    [IsDesignScriptCompatible]
    public class CompasNetwork : IGraphicItem
    {
        #region private members   

        // IronPython mesh object 
        private object pythonNetwork;
        // string representation of this mesh (brought from python)
        private string str;
        // [[x,y,z],[x,y,z],[..]]
        private List<object> vertices;
        private double[][] verticesDouble;

        // [[ptid0, ptid1],[..]]
        private List<object> edgeIndices;
        private int[][] edgesInt;

        #endregion

        #region properties

        // Nothing here yet =).

        #endregion

        #region private methods

        public static string GetPackagePath()
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData) + @"\Dynamo\Dynamo Core\1.3\packages\DynamoCompas\";
        }

        #endregion

        #region public methods

        private CompasNetwork(object _pythonNetwork, string stringRepresentation, List<object> _vertices, List<object> _edgeIndices)
        {
            str = stringRepresentation;
            pythonNetwork = _pythonNetwork;
            vertices = _vertices;
            edgeIndices = _edgeIndices;

            // parse the vertices to double arrays
            // [[x,y,z],[x,y,z],[..]]
            verticesDouble = new double[vertices.Count][];
            int i = 0;
            foreach (List<object> p in vertices)
            {
                double[] triple = new double[3];
                triple[0] = (double)p[0];
                triple[1] = (double)p[1];
                triple[2] = (double)p[2];
                verticesDouble[i++] = triple;
            }

            // [[ptid0, ptid1],[..]]
            edgesInt = new int[edgeIndices.Count][];
            i = 0;
            foreach(List<object> e in edgeIndices)
            {
                int[] edge = new int[2];
                edge[0] = (int)e[0];
                edge[1] = (int)e[1];
                edgesInt[i++] = edge;
            }

        }

        public static CompasNetwork CompasNetworkFromObj(string filePath = null, string IronPythonPath = @"C:\Program Files (x86)\IronPython 2.7")
        {
            string path = GetPackagePath() + @"bin";

            var pySrc =
@"
import sys
sys.path.append(r'C:\Program Files (x86)\IronPython 2.7')
sys.path.append(r'C:\Program Files (x86)\IronPython 2.7\Lib')
sys.path.append(r'C:\Program Files (x86)\IronPython 2.7\DLLs')
sys.path.append(r'C:\Users\JLXMac\AppData\Roaming\Dynamo\Dynamo Core\1.3\packages\DynamoCompas\bin')

import compas
from compas.datastructures.network import Network

# import List class to cast the type
from System.Collections.Generic import *


def NetworkFromObject(filepath):

    # import network
    network = Network.from_obj(filepath)

    # extract network vertices
    xyz = [network.vertex_coordinates(key) for key in network.vertices()]
    vertices = List[object]([List[object]([x, y, z]) for x, y, z in xyz])

    # extract network edges
    key_index = network.key_index()
    edges = [(key_index[u], key_index[v]) for u, v in network.edges()]
    edges = List[object]([List[object](ij) for ij in edges])

    return List[object]([network, str(network), vertices, edges])

";

            if (filePath != null || filePath != "")
            {

                // host python and execute script
                var engine = IronPython.Hosting.Python.CreateEngine();
                var scope = engine.CreateScope();
                engine.Execute(pySrc, scope);

                var NetworkFromObject = scope.GetVariable<Func<string, List<object>>>("NetworkFromObject");
                var networkList = NetworkFromObject(filePath);

                return CompasNetwork.Create(networkList[0], networkList[1] as String, networkList[2] as List<object>, networkList[3] as List<object>);
            }
            return null;
        }

        //[CanUpdatePeriodically(true)]
        [IsVisibleInDynamoLibrary(false)]
        public static CompasNetwork Create(object pythonMesh, string stringRepresentation, List<object> vertices, List<object> indices)
        {
            return new CompasNetwork(pythonMesh, stringRepresentation, vertices, indices);
        }

        #endregion

        #region IGraphicItem interface


        /// <summary>
        /// The Tessellate method in the IGraphicItem interface allows
        /// you to specify what is drawn when dynamo's visualization is
        /// updated.
        /// </summary>
        [IsVisibleInDynamoLibrary(false)]
        public void Tessellate(IRenderPackage package, TessellationParameters parameters)
        {
            // Vertices
            if (verticesDouble != null)
            {
                foreach (double[] p in verticesDouble)
                {
                    package.AddPointVertex(p[0], p[1], p[2]);
                    package.AddPointVertexColor(255, 0, 0, 255);
                }
            }

            if (edgesInt != null)
            {
                foreach (int[] ids in edgesInt)
                {
                    double[] p0 = verticesDouble[ids[0]];
                    double[] p1 = verticesDouble[ids[1]];

                    package.AddLineStripVertexColor(0, 255, 0, 255);
                    package.AddLineStripVertex(p0[0], p0[1], p0[2]);

                    package.AddLineStripVertexColor(0, 0, 255, 255);
                    package.AddLineStripVertex(p1[0], p1[1], p1[2]);

                    package.AddLineStripVertexCount(2);
                }
            }

            //// Edges
            //foreach (List<object> edge in edgeIndices)
            //{
            //    List<object> startPoint = vertices[1] as List<object>;
            //    double startX = (double)startPoint[0];
            //    double startY = (double)startPoint[1];
            //    double startZ = (double)startPoint[2];

            //    List<object> endPoint = vertices[1] as List<object>;
            //    double endX = (double)endPoint[0];
            //    double endY = (double)endPoint[1];
            //    double endZ = (double)endPoint[2];

            //    package.AddLineStripVertex(startX, startY, startZ);
            //    package.AddLineStripVertex(endX, endY, endZ);

            //    package.AddLineStripVertexColor(0, 0, 100, 255);
            //    package.AddLineStripVertexColor(0, 0, 150, 255);

            //    package.AddLineStripVertexCount(1);
            //}

            //internal static void DrawColoredLine(IRenderPackage package, CompasNetworkWrapper network, Point p0, Point p1)
            //{
            //    package.AddLineStripVertex(p0.X, p0.Y, p0.Z);
            //    package.AddLineStripVertex(p1.X, p1.Y, p1.Z);

            //    byte blue0 = (byte)Math.Round(255 * ((p0.Z - network.minZ) / network.dz));
            //    byte blue1 = (byte)Math.Round(255 * ((p1.Z - network.minZ) / network.dz));

            //    package.AddLineStripVertexColor(0, 0, blue0, 255);
            //    package.AddLineStripVertexColor(0, 0, blue1, 255);

            //    // Specify line segments by adding a line vertex count.
            //    // Ex. The above line has two vertices, so we add a line
            //    // vertex count of 2. If we had tessellated a curve with n
            //    // vertices, we would add a line vertex count of n.
            //    package.AddLineStripVertexCount(2);
            //}
        }

        #endregion

        public object ToPythonNetwork()
        {
            return this.pythonNetwork;
        }

        public override string ToString()
        {
            //return string.Format("{0}", this.str);
            return this.str;
        }
    }
}
