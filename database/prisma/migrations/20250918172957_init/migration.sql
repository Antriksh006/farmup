-- CreateTable
CREATE TABLE "public"."User" (
    "state" TEXT NOT NULL,
    "market" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "crop" TEXT NOT NULL,
    "price" TEXT NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("state")
);
