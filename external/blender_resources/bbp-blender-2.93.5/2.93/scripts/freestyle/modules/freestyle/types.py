# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

"""
This module contains core classes of the Freestyle Python API,
including data types of view map components (0D and 1D elements), base
classes for user-defined line stylization rules (predicates,
functions, chaining iterators, and stroke shaders), and operators.

Class hierarchy:

- :class:`BBox`
- :class:`BinaryPredicate0D`
- :class:`BinaryPredicate1D`
- :class:`Id`
- :class:`Interface0D`

  - :class:`CurvePoint`

    - :class:`StrokeVertex`

  - :class:`SVertex`
  - :class:`ViewVertex`

    - :class:`NonTVertex`
    - :class:`TVertex`

- :class:`Interface1D`

  - :class:`Curve`

    - :class:`Chain`

  - :class:`FEdge`

    - :class:`FEdgeSharp`
    - :class:`FEdgeSmooth`

  - :class:`Stroke`
  - :class:`ViewEdge`

- :class:`Iterator`

  - :class:`AdjacencyIterator`
  - :class:`CurvePointIterator`
  - :class:`Interface0DIterator`
  - :class:`SVertexIterator`
  - :class:`StrokeVertexIterator`
  - :class:`ViewEdgeIterator`

    - :class:`ChainingIterator`

  - :class:`orientedViewEdgeIterator`

- :class:`Material`
- :class:`Noise`
- :class:`Operators`
- :class:`SShape`
- :class:`StrokeAttribute`
- :class:`StrokeShader`
- :class:`UnaryFunction0D`

  - :class:`UnaryFunction0DDouble`
  - :class:`UnaryFunction0DEdgeNature`
  - :class:`UnaryFunction0DFloat`
  - :class:`UnaryFunction0DId`
  - :class:`UnaryFunction0DMaterial`
  - :class:`UnaryFunction0DUnsigned`
  - :class:`UnaryFunction0DVec2f`
  - :class:`UnaryFunction0DVec3f`
  - :class:`UnaryFunction0DVectorViewShape`
  - :class:`UnaryFunction0DViewShape`

- :class:`UnaryFunction1D`

  - :class:`UnaryFunction1DDouble`
  - :class:`UnaryFunction1DEdgeNature`
  - :class:`UnaryFunction1DFloat`
  - :class:`UnaryFunction1DUnsigned`
  - :class:`UnaryFunction1DVec2f`
  - :class:`UnaryFunction1DVec3f`
  - :class:`UnaryFunction1DVectorViewShape`
  - :class:`UnaryFunction1DVoid`

- :class:`UnaryPredicate0D`
- :class:`UnaryPredicate1D`
- :class:`ViewMap`
- :class:`ViewShape`
- :class:`IntegrationType`
- :class:`MediumType`
- :class:`Nature`
"""


# module members
