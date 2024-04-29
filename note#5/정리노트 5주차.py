#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
from dezero import Variable, Function
from dezero.utils import plot_dot_graph
import math



# In[5]:


==24==


# In[ ]:


Dezero의 연산자 지원) 사칙연산과 제곱연산을 지원
최적화 문제의 테스트 함수) Sphere함수, mathyas함수, Goldstein-Price함수의 미분 수행
Dezero 기능) 1. 역전파 2.코드 작성 가능(by연산자 오버로드) 3. 일반적인 프로그래밍을 미분 가능


# In[ ]:


#Sphere함수
z = x^2+y^2
(x,y)=(1.0, 1.0)인 경우 미분 결과는 (2.0,2.0)


# In[ ]:


#mathyas함수
z = 0.26(x^2+y^2) - 0.48xy
(x,y)=(1.0, 1.0)인 경우 미분 결과는 (0.04,0.04)


# In[ ]:


#Goldstein-Price함수
f(x,y) = [1+(x+y+1)^2(19-14x+3x^2-14y+6xy+3y^2)]
        [30+(2x-3y)^2(18-32x+12x^2+48y-36xy+27y^2)]
(x,y)=(1.0, 1.0)인 경우 미분 결과는 (-5376.0, 8064.0)


# In[2]:


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


# In[3]:


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


# In[6]:


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


# In[8]:


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)


# In[9]:


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)


# In[10]:


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)


# In[ ]:


==25==


# In[ ]:


계산 그래프 시각화 필요성)
직접 그래프가 만들어지는 전모 확인
문제 발생시 원인파악 쉬움
신경망구조 시각적 전달 용도


# In[ ]:


==26==


# In[ ]:


역전파는 출력 변수를 기점으로 역방향으로 모든 변수와 함수를 추적


# In[ ]:


#_dot_var함수
get_dot_graph 함수 전용으로 로컬에서만 사용함
Variable인스턴스를 건네면 인스턴스 내용을 DOT언어로 작성된 문자열로 변환
파이썬 내장 함수인 id사용
format메서드 문자열의{}부분을 인수로 건넨 객체로 차례로 바꿔줌


# In[ ]:


#_dot_func함수
get_dot_graph 함수 전용으로 로컬에서만 사용함
dezero함수를 dot언어로 기술
dezero함수는 function클래스를 상속하고, inputs와 outputs라는 인스턴스 변수 가짐


# In[ ]:


#get_dot_graph함수
backward메서드는 미분값을 전파 -> 미분 대신 DOT언어로 기술한 문자열 txt추가
역전파는 노드를 따라가는 순서가 중요하여 함수에 generation 정수값 부여
노드를 추적하는 순서 필요없어 generaion값으로 정력하는 코드는 주석처리


# In[12]:


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


# In[13]:


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()


# In[14]:


x.name = 'x'
y.name = 'y'
z.name = 'z'
plot_dot_graph(z, verbose=False, to_file='goldstein.png')


# In[ ]:


==27==


# In[ ]:


테일러급수) 어떤 미지의 함수를 동일한 미분계수를 갖는 어떤 다항함수로 근사
미분을 n번 계속 진행하고 어느 시점에서 중단하면 f(x)의 값을 근사
항이 많아 질수록 근사의 정확도가 높음


# In[ ]:


테일러급수 식에 따라 sin함수를 코드로 구현 > 구현한 sin함수와 거의 같은 결과를 얻음.
오차는 무시할 정도로 작고, threshold값을 줄이면 오차를 더 줄일 수 있다.


# In[18]:


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


# In[19]:


def sin(x):
    return Sin()(x)


# In[20]:


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()
print('--- original sin ---')
print(y.data)
print(x.grad)


# In[21]:


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


# In[22]:


x = Variable(np.array(np.pi / 4))
y = my_sin(x)  # , threshold=1e-150)
y.backward()
print('--- approximate sin ---')
print(y.data)
print(x.grad)


# In[23]:


x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin.png')


# In[ ]:


==28==


# In[ ]:


함수의 최적화와 최적화 계산을 위해 미분의 역할은 커진다.
최적화: 어떤 함수가 주어졌을때 최솟값(최댓값)을 반환하는 입력(함수의 인수)를 찾는 일
신경망 학습 목표도 최적화 문제에 속함


# In[ ]:


#포젠브록 함수
y = 100(x1-x0^2)^2+(1-x0)^2
a,b가 정수 일때 f(x0,x1)=b(x1-x0^2)^2 + (a-x0)^2
a = 1, b = 100으로 설정하여 벤치마크
포물선 모양으로 길게 뻗은 골짜기 보이는 형태

최적화) 출력이 최소가 되는 x0와 x1찾기
최솟값이 되는 지점은 (x0, x1) = (1,1)


##경사하강법
복잡한 형상의 함수 > 기울기가 가리키는 방향에 반드시 최솟값 존재x
기울기는 함수의 출력을 가장 크게 하는 방향에 나타남
좋은 초깃값은 경사하강법을 목적지까지 효율적으로 도달


>> 최솟값 찾기
1. 기울기 방향에 마이너스를 곱한 방향으로 이동
2. iters는 반복횟수, Ir은 학습률
3. cleargrad 메서드
4. (x0,x1)값이 갱신됨
>>>>경사하강법은 로젠브록 함수 같이 골짜기가 길게 뻗은 함수에 대응 못함


# In[25]:


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


# In[26]:


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000


# In[27]:


for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad


# In[ ]:


==29==


# In[ ]:


##뉴턴 방법
더 적은 단계로 최적의 결과를 얻을 수 있다.
테일러 급수에 따라 어떤 점 a를 기점으로 f를 x의 다항식으로 나타낼 수 있음
2차 미분의 정보도를 이용한다.
속도와 가속도 정보를 사용하여 효율적으로 탐색 기대
>> 1차 미분은 역전파로 구하고 2차 미분은 수동으로 코딩해 구해야하지만 7회만의 갱신으로 최솟값에 도달.


# In[28]:


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


# In[29]:


def gx2(x):
    return 12 * x ** 2 - 4


# In[30]:


x = Variable(np.array(2.0))
iters = 10


# In[31]:


for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)

