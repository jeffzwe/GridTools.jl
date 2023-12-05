

Benchmarks

using BenchmarkTools
using Profile
#########################################################################################################

add3(a, b, c) = a.+b.+c
add3(rand(10,10), rand(10,10), rand(10,10))
add3(Field((Cell, K), rand(10,10)),Field((Cell, K), rand(10,10)),Field((Cell, K), rand(10,10)))

function prof()
    for i in [1000, 10000, 100000, 1000000]
        k =100
        # Profile.clear()
        # for _ in 0:10000
        #     a=rand(i,k)
        #     b=rand(i,k)
        #     c=rand(i,k)
        #     @profile add3(a, b, c)
        # end
        # print("Plain arrays:")
        # Profile.print()

        Profile.clear()
        for j in 0:250
            a=Field((Cell, K), rand(i,k))
            b=Field((Cell, K), rand(i,k))
            c=Field((Cell, K), rand(i,k))
            @profile add3(a, b, c)
        end
        print("Fields:")
        Profile.print()
        
        break
    end
end

function bench()
    for i in [1000, 10000, 100000, 1000000]
        k =100
        @btime (unos .+ dos .+ tres) setup=(
            unos=Field((Cell, K), rand($i,$k)); 
            dos=Field((Cell, K), rand($i,$k)); 
            tres=Field((Cell, K), rand($i,$k)))

        # @btime (a .+ b .+ c) setup=(
        #     a=rand($i,$k); 
        #     b=rand($i,$k); 
        #     c=rand($i,$k))

        # @btime (a + b - c) setup=(
        # a=rand($i,$k); 
        # b=rand($i,$k); 
        # c=rand($i,$k))
    end
end


#########################################################################################################




mask = Field((Cell, K), rand(Bool,10,10))
aa = Field((Cell, K), rand(10,10))
bb = Field((Cell, K), rand(10,10))
cc = Field((Cell, K), rand(10,10))
dd = Field((Cell, K), rand(10,10))
ee = Field((Cell, K), rand(10,10))
ff = Field((Cell, K), rand(10,10))

where(mask, (aa,(bb,cc)), (dd, (ee,ff)))
whereit(mask, (aa,(bb,cc)), (dd, (ee,ff)))

 
function bench()
    for i in [1000, 10000, 100000]
        k =100

        println("where $i")

        @btime where(mask, (a,(b,c)), (d,(e,f))) setup=(
            mask=Field((Cell, K), rand(Bool, $i,$k)); 
            a=Field((Cell, K), rand($i,$k)); 
            b=Field((Cell, K), rand($i,$k));
            c=Field((Cell, K), rand($i,$k));
            d=Field((Cell, K), rand($i,$k));
            e=Field((Cell, K), rand($i,$k));
            f=Field((Cell, K), rand($i,$k));)

        println("whereit $i")

        @btime whereit(mask, (a,(b,c)), (d,(e,f))) setup=(
            mask=Field((Cell, K), rand(Bool, $i,$k)); 
            a=Field((Cell, K), rand($i,$k)); 
            b=Field((Cell, K), rand($i,$k));
            c=Field((Cell, K), rand($i,$k));
            d=Field((Cell, K), rand($i,$k));
            e=Field((Cell, K), rand($i,$k));
            f=Field((Cell, K), rand($i,$k));)

    end
end

