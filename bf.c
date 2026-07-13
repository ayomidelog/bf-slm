/*
 * bf.c — Fast C Brainfuck interpreter
 * gcc -O3 -o bf bf.c && echo -n "de" | ./bf slm_pidgin.bf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAPE_SIZE 120000
#define MAX_PROGRAM 800000

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program.bf>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }

    int *prog = malloc(MAX_PROGRAM * sizeof(int));
    int *bracket = malloc(MAX_PROGRAM * sizeof(int));
    int *bstack = malloc(MAX_PROGRAM * sizeof(int));
    unsigned char *tape = calloc(TAPE_SIZE, 1);

    if (!prog || !bracket || !bstack || !tape) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    int len = 0;
    int c;
    int in_comment = 0;
    while ((c = fgetc(f)) != EOF && len < MAX_PROGRAM) {
        if (c == ';') {
            in_comment = 1;
            continue;
        }
        if (c == '\n' || c == '\r') {
            in_comment = 0;
            continue;
        }
        if (in_comment) continue;
        switch (c) {
            case '>': case '<': case '+': case '-':
            case '.': case ',': case '[': case ']':
                prog[len++] = c;
                break;
            default:
                break;
        }
    }
    fclose(f);

    fprintf(stderr, "Loaded %d instructions\n", len);

    /* Precompute bracket map */
    int sp = 0;
    for (int i = 0; i < len; i++) {
        if (prog[i] == '[') {
            bstack[sp++] = i;
        } else if (prog[i] == ']') {
            if (sp == 0) {
                fprintf(stderr, "Unmatched ']' at %d\n", i);
                return 1;
            }
            int j = bstack[--sp];
            bracket[j] = i;
            bracket[i] = j;
        }
    }

    int p = 0;
    int ip = 0;

    while (ip < len) {
        switch (prog[ip]) {
            case '>': p++; break;
            case '<': p--; break;
            case '+': tape[p]++; break;
            case '-': tape[p]--; break;
            case '.': putchar(tape[p]); break;
            case ',': {
                int b = getchar();
                tape[p] = (b == EOF) ? 0 : (unsigned char)b;
                break;
            }
            case '[':
                if (tape[p] == 0) ip = bracket[ip];
                break;
            case ']':
                if (tape[p] != 0) ip = bracket[ip];
                break;
        }
        ip++;
    }

    free(prog);
    free(bracket);
    free(bstack);
    free(tape);
    return 0;
}
